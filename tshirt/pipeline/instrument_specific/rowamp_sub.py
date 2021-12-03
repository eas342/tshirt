import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from ..utils import get_baseDir
import os
import pdb
from copy import deepcopy

def do_backsub(img,photObj=None,amplifiers=4,saveDiagnostics=False):
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
    
    saveDiagnostics: bool
        Save diagnostic files?
    """
    
    ## make a mask
    if photObj is None:
        useMask = np.ones_like(img,dtype=bool)
    else:
        y, x = np.mgrid[0:img.shape[0],0:img.shape[1]]
        
        xcen = photObj.srcApertures.positions[:,0]
        ycen = photObj.srcApertures.positions[:,1]
        
        useMask = np.ones_like(img,dtype=bool)
        for srcInd in np.arange(photObj.nsrc):
            r = np.sqrt((x - xcen)**2 + (y - ycen)**2)
            srcPts = r < photObj.param['backStart']
            useMask[srcPts] = False
    
    
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
            ## find the median background along a row
            medVals = np.nanmedian(masked_img[:,ampStarts[amp]:ampEnds[amp]],axis=1)
            ## tile this to make a constant model
            tiled_med = np.tile(medVals, [ampWidth,1]).T
            ## put the results in the model image
            modelimg[:,ampStarts[amp]:ampEnds[amp]] = tiled_med
        
    elif amplifiers == 1:
        medVals = np.nanmedian(masked_img,axis=1)
        ## tile this to make a constant model
        tiled_med = np.tile(medVals, [img.shape[1],1]).T
        ## put the results in the model image
        modelimg = tiled_med
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
    
    