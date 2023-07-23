import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
import warnings
try:
    from ..utils import get_baseDir
except ImportError as err1:
    warnings.warn("Could not import baseDir. Will save diagnostics to . ")
    def get_baseDir():
        return "."
import os
import pdb
from copy import deepcopy
from scipy.signal import savgol_filter, savgol_coeffs
from astropy.convolution import convolve
import celerite2


def do_even_odd(thisAmp):
    """
    Do an even-odd correction for a given amplifier
    If only one amplifier is used, it can be the whole image
    """
    even_odd_model = np.zeros_like(thisAmp)
    
    even_offset = np.nanmedian(thisAmp[:,0::2])
    even_odd_model[:,0::2] = even_offset
    odd_offset = np.nanmedian(thisAmp[:,1::2])
    even_odd_model[:,1::2] = odd_offset
    
    return thisAmp - even_odd_model, even_odd_model

def col_by_col(thisAmp):
    """
    Do a column-by-column slow read correction
    instead of an even/odd correction
    """
    colMedian = np.nanmedian(thisAmp,axis=0)
    slowread_model = np.tile(colMedian,[thisAmp.shape[0],1])
    return thisAmp - slowread_model, slowread_model

def make_time2D(img,amplifiers=4):
    nY, nX = img.shape
    if amplifiers == 4:
        numColsPerAmp  = 512
    elif amplifiers == 1:
        numColsPerAmp = nX
    else:
        raise Exception("Unrecognized number of amps")
    time1Damp_over = np.arange((numColsPerAmp + 12) * nY)
    time2Damp_over = time_wrap(time1Damp_over,nY,(numColsPerAmp+12))
    time2Damp = time2Damp_over[:,0:numColsPerAmp]

    if amplifiers == 1:
        time2D = time2Damp
    elif amplifiers == 4:
        time2D = np.zeros([nY,2048]) * np.nan
        time2D[:,0:512] = time2Damp
        time2D[:,512:1024] = time2Damp[:,::-1]
        time2D[:,1024:1536] = time2Damp
        time2D[:,1536:2048] = time2Damp[:,::-1]
    else:
        raise Exception("Unrecognized number of amps")

    return time2D * 10e-6

def time_wrap(img1D,rows,columns,amp=0):
    #return np.reshape(img1D,[rows,columns],order='F')
    img2D = np.reshape(img1D,[rows,columns],order='C')
    if (amp == 0) | (amp == 2):
        return img2D
    else:
        return img2D[:,::-1]

def time_unwrap(img2D,amp=0):
    if (amp == 0) | (amp == 2):
        return img2D.ravel()
    else:
        return img2D[:,::-1].ravel()

def show_gp_psd():
    """
    Show the power spectral density function for the Gaussian process
    """


    #minfreq, maxfreq = 1.0 / np.max(times), 1.0 / (times[1] - times[0])
    minfreq, maxfreq = 1e-4, 1e2
    freq = np.logspace(np.log10(minfreq),np.log10(maxfreq), 9000)
    omega = 2 * np.pi * freq

    gp_comb, gp_list = make_gp()

    for ind,oneGP in enumerate(gp_list):
        plt.loglog(freq, oneGP.kernel.get_psd(omega),label="SHOTerm {}".format(ind+1))
    plt.loglog(freq, gp_comb.kernel.get_psd(omega),label="Combined")

    plt.plot(freq,1./freq * 7.,label='1/f power law',linestyle='dashed')
    plt.legend()
    plt.ylim(1e-3,1e5)
    plt.xlabel("Frequency (1/ms)")
    plt.ylabel("Power")
    plt.show()

def make_gp(sigma_w=7.):
    """
    Make a Gaussian Process Kernel for 1/f corrections

    Parameters
    -----------
    sigma_w: float
        White noise error in DN
    """
    terms_list = []
    gp_list = []

    rho_list = [1000, 100, 10, 1, 0.1, 0.01]
    for ind,rho in enumerate(rho_list):
        oneTerm = celerite2.terms.SHOTerm(sigma=sigma_w, rho=rho, tau=0.1 * rho)
        if ind == 0:
            kernel_comb = oneTerm
        else:
            kernel_comb = kernel_comb + oneTerm
        gp = celerite2.GaussianProcess(oneTerm, mean=0)
        gp_list.append(gp)

    gp_comb = celerite2.GaussianProcess(kernel_comb, mean=0.0)
    return gp_comb, gp_list

def gp_predict(t,f,gp_model,sigma_w=7.,nsmoothKern=501):
    pts_finite = np.isfinite(f)
    pts_non_outlier = (np.abs(f - np.nanmedian(f)) < 100.)
    pts = pts_finite & pts_non_outlier
    gp_model.compute(t[pts],yerr=sigma_w * np.ones(np.sum(pts)))
    mu_gp = gp_model.predict(f[pts], t=t, return_var=False)

    kernel_for_gp_smoothing = savgol_coeffs(nsmoothKern,3)

    gp_smoothed = convolve(mu_gp, kernel_for_gp_smoothing, boundary='extend')
    #pdb.set_trace()
    #plt.plot(t,f,'.'); plt.plot(t,gp_smoothed); plt.ylim(-100,100); plt.show()
    return gp_smoothed

def do_backsub(img,photObj=None,amplifiers=4,saveDiagnostics=False,
               evenOdd=True,activePixMask=None,backgMask=None,
               grismr=False,returnFastSlow=False,
               colByCol=False,smoothSlowDir=None,
               badRowsAllowed=0,
               GROEBA=False,GPnsmoothKern=None,
               showGP1D=False):
    """
    Do background subtraction amplifier-by-amplifier, row-by-row around the sources
    
    Parameters
    ---------------
    img: numpy 2D array
        The input uncorrected image
    
    photObj: (optional) a tshirt photometry pipeline object
        If supplied, it will use the background apertures to mask out sources
    
    amplifiers: int
        How many outputamplifiers are used? 4 for NIRCam stripe mode and 1 is for 1 output amplifier
    
    evenOdd: bool
        Remove the even and odd offsets before doing row-by-row medians? 
    
    colByCol: bool
        Do column-by-column subtraction. This will supercede the even/odd correction.
    
    saveDiagnostics: bool
        Save diagnostic files?
               
    activePixMask: numpy 2D array or None
        Mask of reference pixels to ignore (optionally). Pixels that are False
               will be ignored in the background estimation, and pixels that are 
               true will be kept in the background estimation. If refpixMask is None
               no extra points will be masked
    
    badRowsAllowed: int
        Number of bad rows to accept and still do fast-read correction
            (ie. if not enough background and/or ref pixels, 
            no fast-read correction can be done)

    backgMask: numpy 2D array or None
        Mask for the background. Pixels that are False will be ignored in
               row-by-row background estimation. Pixels that are true will
               be kept for the background estimation.
               
    grismr: bool
        Is this NIRCam GRISMR data? Special treatment is needed for NIRCam
               GRISMR, where the spectra run through multiple amplifiers
    
    returnFastSlow: bool
        Return both the fast and slow read models?
    
    smoothSlowDir: None or float
        Length of the smoothing kernel to apply along the fast read direction 
        If None, no smoothing is applied
    
    GROEBA: bool
        Use Gaussian-Process row-by-row, odd/even by amplifier subtraction?
        This will use a Gaussian process to do 1/f noise interpolation

    GPnsmoothKern: int or None
        (only if GROEBA==True). Specify the window length for a Savgol smoothing
        filter for the GP. Otherwise it is chosen automatically

    showGP1D: bool
        (only if GROEBA==True). Plot the GP prediction in 1D
    """
    
    ## Npix Threshold
    ## this many pixels must be in a row in order to do an median
    ## otherwise, it attempts to interpolate from other amplifiers
    ## this only applies when noutputs=4
    Npix_threshold = 3
    
    ## Start by including all pixels
    useMask = np.ones_like(img,dtype=bool)
    
    ## make a mask with Tshirt object
    if photObj is not None:
        y, x = np.mgrid[0:img.shape[0],0:img.shape[1]]
        
        xcen = photObj.srcApertures.positions[:,0]
        ycen = photObj.srcApertures.positions[:,1]
        
        useMask = np.ones_like(img,dtype=bool)
        for srcInd in np.arange(photObj.nsrc):
            r = np.sqrt((x - xcen)**2 + (y - ycen)**2)
            srcPts = r < photObj.param['backStart']
            useMask[srcPts] = False
    if backgMask is not None:
        useMask = useMask & backgMask
        
    
    
    
    
    ## only include active pixels
    if activePixMask is None:
        activeMask = np.ones_like(img,dtype=bool)
    else:
        activeMask = activePixMask
    
    useMask = useMask & activeMask
    
    ## let nanmedian do the work of masking
    maskedPts = (useMask == False)
    masked_img = deepcopy(img)
    masked_img[maskedPts] = np.nan
    
    outimg = np.zeros_like(img)
    slowread_model = np.zeros_like(img)
    fastread_model = np.zeros_like(img)

    if GROEBA == True:
        sigma_w = 7.
        gp_comb, gp_list = make_gp(sigma_w=sigma_w)
        time2D = make_time2D(img,amplifiers=amplifiers)

    rowsAmp, totcols = img.shape
    if amplifiers == 4:
        colsAmp = 512
    else:
        colsAmp = totcols

    ## mask to keep track of which amplifiers have enough pixels to use
    ## in NIRCam GRISMR spectroscopy, the spectra cut across rows
    ## so these amplifiers' 1/f noise has to interpolated from other 
    ## amps
    amp_check_mask = np.ones(amplifiers,dtype=bool)
    if amplifiers == 4:
        ## list where the amplifiers are
        ampStarts = [0,512,1024,1536]
        ampEnds = [512,1024,1536,2048]
        ampWidth = 512
    elif amplifiers == 1:
        ampStarts = [0]
        ampEnds = [colsAmp]
        ampWidth = colsAmp
    else:
        raise Exception("{} amplifiers not implemented".format(amplifiers))
    
    ## loop through amps
    
    for amp in np.arange(amplifiers):
        if colByCol == True:
            thisAmp, colbycol_model = col_by_col(masked_img[:,ampStarts[amp]:ampEnds[amp]])
            slowread_model[:,ampStarts[amp]:ampEnds[amp]] = colbycol_model
        elif evenOdd == True:
            thisAmp, even_odd_model = do_even_odd(masked_img[:,ampStarts[amp]:ampEnds[amp]])
            slowread_model[:,ampStarts[amp]:ampEnds[amp]] = even_odd_model
        else:
            thisAmp = masked_img[:,ampStarts[amp]:ampEnds[amp]]
            ## even if not doing an even/odd correction, still do an overall median
            slowread_model[:,ampStarts[amp]:ampEnds[amp]] = np.nanmedian(thisAmp)
            thisAmp = thisAmp - slowread_model[:,ampStarts[amp]:ampEnds[amp]]
        
        
        ## check if the mask leaves too few pixels in this amplifier
        ## in that case
        pxPerRow = np.sum(np.isfinite(thisAmp),axis=1)
        bad_rows = np.sum(pxPerRow <= Npix_threshold)
        
        if bad_rows <= badRowsAllowed:
            if GROEBA == True:
                t = time_unwrap(time2D[:,ampStarts[amp]:ampEnds[amp]],amp=amp)
                f = time_unwrap(thisAmp,amp=amp)
                if GPnsmoothKern is None:
                    if np.median(pxPerRow) < 6:
                        ## Smooth more for small numbers of pixels like when using refpix
                        nsmoothKern = 15001
                    else:
                        nsmoothKern = 2001
                
                mu_gp1D = gp_predict(t,f,gp_comb,sigma_w=sigma_w,nsmoothKern=GPnsmoothKern)
                if showGP1D == True:
                    plt.plot(t,f); plt.plot(t,mu_gp1D)
                    plt.show()
                    pdb.set_trace()
                fastread_model[:,ampStarts[amp]:ampEnds[amp]] = time_wrap(mu_gp1D,
                                                                          columns=colsAmp,
                                                                          rows=rowsAmp,
                                                                          amp=amp)
            else:
                medVals = np.nanmedian(thisAmp,axis=1)
                if smoothSlowDir is not None:
                    medVals = savgol_filter(medVals,smoothSlowDir,3)
                
                ## tile this to make a model across the fast-read direction
                fastread_model[:,ampStarts[amp]:ampEnds[amp]] = np.tile(medVals, [ampWidth,1]).T
            
            if np.median(pxPerRow) < 6:
                amp_check_mask[amp] = False ## For amplifiers where there are few pixels,
                ## count this as "unchecked" to average other amplifier's information
            else:
                amp_check_mask[amp] = True
        else:
            amp_check_mask[amp] = False
            fastread_model[:,ampStarts[amp]:ampEnds[amp]] = 0.0
        

    
    ## If there is at least 1 good amp and 1 bad amp,
    ## the good amp can be used to
    ## estimate the 1/f noise in "bad" amps where some rows have
    ## no background pixels that can used
    ngood_amps = np.sum(amp_check_mask)
    ## only do this if we have >=1 good and >=1 bad amp
    if (ngood_amps >= 1) & (ngood_amps < amplifiers):
        rowModel = np.nanmean(fastread_model,axis=1)
        tiled_avg = np.tile(rowModel,[ampWidth,1]).T
        for amp in np.arange(4):
            if amp_check_mask[amp] == False:
                fastread_model[:,ampStarts[amp]:ampEnds[amp]] = tiled_avg
        

        
    
        

    
    ## put the results in the model image
    modelimg = slowread_model + fastread_model
    
    outimg = img - modelimg
    if saveDiagnostics == True:
        if photObj is None:
            outPrefix = 'unnamed'
        else:
            outPrefix = photObj.dataFileDescrip
        
        diag_dir = os.path.join(get_baseDir(),'diagnostics','rowamp_sub')
        descrips = ['orig','mask','slowread_model','slowread_sub',
                    'fastread_model','model','subtracted']
        diag_imgs = [img,masked_img,slowread_model,img-slowread_model,
                     fastread_model,modelimg,outimg]
        for outInd,outDescrip in enumerate(descrips):
            out_path = os.path.join(diag_dir,"{}_{}.fits".format(outPrefix,outDescrip))
            outHDU = fits.PrimaryHDU(diag_imgs[outInd])
            outHDU.writeto(out_path,overwrite=True)
            
    if returnFastSlow == True:
        return outimg, slowread_model, fastread_model
    else:
        return outimg, modelimg
    
    