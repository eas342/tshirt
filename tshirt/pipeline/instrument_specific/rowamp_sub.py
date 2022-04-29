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

def do_backsub(img,photObj=None,amplifiers=4,saveDiagnostics=False,
               evenOdd=True,activePixMask=None,backgMask=None,
               grismr=False):
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
    
    saveDiagnostics: bool
        Save diagnostic files?
               
    activePixMask: numpy 2D array or None
        Mask of reference pixels to ignore (optionally). Pixels that are False
               will be ignored in the background estimation, and pixels that are 
               true will be kept in the background estimation. If refpixMask is None
               no extra points will be masked
    
    backgMask: numpy 2D array or None
        Mask for the background. Pixels that are False will be ignored in
               row-by-row background estimation. Pixels that are true will
               be kept for the background estimation.
               
    grismr: bool
        Is this NIRCam GRISMR data? Special treatment is needed for NIRCam
               GRISMR, where the spectra run through multiple amplifiers
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
    
    if amplifiers == 4:
        ## mask to keep track of which amplifiers have enough pixels to use
        ## in NIRCam GRISMR spectroscopy, the spectra cut across rows
        ## so these amplifiers' 1/f noise has to interpolated from other 
        ## amps
        amp_check_mask = np.ones(amplifiers,dtype=bool)
        
        ## list where the amplifiers are
        ampStarts = [0,512,1024,1536]
        ampEnds = [512,1024,1536,2048]
        ampWidth = 512
        
        ## loop through amps
        
        for amp in np.arange(4):
            if evenOdd == True:
                thisAmp, even_odd_model = do_even_odd(masked_img[:,ampStarts[amp]:ampEnds[amp]])
                slowread_model[:,ampStarts[amp]:ampEnds[amp]] = even_odd_model
            else:
                thisAmp = masked_img[:,ampStarts[amp]:ampEnds[amp]]
                ## even if not doing an even/odd correction, still do an overall median
                slowread_model[:,ampStarts[amp]:ampEnds[amp]] = np.nanmedian(thisAmp)
                thisAmp = thisAmp - slowread_model[:,ampStarts[amp]:ampEnds[amp]]
            
            
            ## check if the mask leaves too few pixels in this amplifier
            ## in that case
            bad_rows = np.sum(np.sum(np.isfinite(thisAmp),axis=1) <= Npix_threshold)
            if bad_rows == 0:
                medVals = np.nanmedian(thisAmp,axis=1)
                ## tile this to make a model across the fast-read direction
                fastread_model[:,ampStarts[amp]:ampEnds[amp]] = np.tile(medVals, [ampWidth,1]).T
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
        

        
    elif amplifiers == 1:
        if evenOdd == True:
            thisAmp,even_odd_model = do_even_odd(masked_img)
            slowread_model = even_odd_model
        else:
            
            slowread_model[:,:] = np.median(thisAmp)
            thisAmp = masked_img - slowread_model
        
        medVals = np.nanmedian(thisAmp,axis=1)
        ## tile this to make a constant model
        tiled_med = np.tile(medVals, [img.shape[1],1]).T
        ## put the results in the model image
        fastread_model[:,:] = tiled_med
        
    else:
        raise Exception("{} amplifiers not implemented".format(amplifiers))
    
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
            
        
    return outimg, modelimg
    
    