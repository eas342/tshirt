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
               evenOdd=True,activePixMask=None,backgMask=None):
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
    """
    
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
    modelimg = np.zeros_like(img)
    
    if amplifiers == 4:
        ## list where the amplifiers are
        ampStarts = [0,512,1024,1536]
        ampEnds = [512,1024,1536,2048]
        ampWidth = 512
        ## loop through
        for amp in np.arange(4):
            if evenOdd == True:
                thisAmp, even_odd_model = do_even_odd(masked_img[:,ampStarts[amp]:ampEnds[amp]])
            else:
                thisAmp = masked_img[:,ampStarts[amp]:ampEnds[amp]]
                even_odd_model = np.zeros_like(thisAmp)
            
            ## find the median background along a row
            medVals = np.nanmedian(thisAmp,axis=1)
            ## tile this to make a constant model
            tiled_med = np.tile(medVals, [ampWidth,1]).T
            ## put the results in the model image
            modelimg[:,ampStarts[amp]:ampEnds[amp]] = tiled_med + even_odd_model
        
    elif amplifiers == 1:
        if evenOdd == True:
            thisAmp,even_odd_model = do_even_odd(masked_img)
        else:
            thisAmp = masked_img
            even_odd_model = np.zeros_like(thisAmp)
        
        medVals = np.nanmedian(thisAmp,axis=1)
        ## tile this to make a constant model
        tiled_med = np.tile(medVals, [img.shape[1],1]).T
        ## put the results in the model image
        modelimg = tiled_med + even_odd_model
    else:
        raise Exception("{} amplifiers not implemented".format(amplifiers))
    
    outimg = img - modelimg
    if saveDiagnostics == True:
        if photObj is None:
            outPrefix = 'unnamed'
        else:
            outPrefix = photObj.dataFileDescrip
        
        diag_dir = os.path.join(get_baseDir(),'diagnostics','rowamp_sub')
        descrips = ['orig','mask','model','subtracted']
        diag_imgs = [img,masked_img,modelimg,outimg]
        for outInd,outDescrip in enumerate(descrips):
            out_path = os.path.join(diag_dir,"{}_{}.fits".format(outPrefix,outDescrip))
            outHDU = fits.PrimaryHDU(diag_imgs[outInd])
            outHDU.writeto(out_path,overwrite=True)
            
        
    return outimg, modelimg
    
    