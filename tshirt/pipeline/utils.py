#import phot_pipeline
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits, ascii
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy
import os
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
import warnings

def robust_statistics(data,method='robust mean',nsig=10):
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    if method == 'median':
        oneStatistic = median_val
        err = mad / np.sqrt(np.sum(np.isfinite(data)))
    elif method == 'robust mean':
        goodp = np.abs(data - median_val) < (nsig * mad)
        oneStatistic = np.mean(data[goodp])
        err = mad / np.sqrt(np.sum(goodp))
    else:
        raise Exception("Unrecognized statistic {}".format(method))
    
    return oneStatistic, err

def robust_poly(x,y,polyord,sigreject=3.0,iteration=3,useSpline=False,knots=None,
                preScreen=False,plotEachStep=False):
    """
    Fit a function (with sigma rejection) to a curve
    
    Parameters
    -----------
    x: numpy array
        Independent variable
    y: numpy array
        Dependent variable
    polyord: int
        order of the fit (number of terms). polyord=1 is a linear fit,
        2 is a quadratic, etc.
    sigreject: float
        The 'sigma' rejection level in terms of median absolute deviations
    useSpline: bool
        Do a spline fit?
    knots: int or None
        How many knots to use if doing a spline fit
    preScreen: bool
        Pre-screen by removing outliers from the median (which might fail for large slopes)
    plotEachStep: bool
        Plot each step of the fitting?
    
    
    Example
    --------------
    .. code-block:: python
    
        import numpy as np
        from tshirt.pipeline import phot_pipeline
        import matplotlib.pyplot as plt
        
        x = np.arange(30)
        y = np.random.randn(30) + x
        y[2] = 80 ## an outlier
        polyfit = phot_pipeline.robust_poly(x,y,1)
        ymodel = np.polyval(polyfit,x)
        plt.plot(x,y,'o',label='input')
        plt.plot(x,ymodel,label='fit')
        plt.show()
        
    """
    finitep = np.isfinite(y) & np.isfinite(x)
    
    if preScreen == True:
        resid = np.abs(y - np.nanmedian(y))
        madev = np.nanmedian(resid)
        goodp = np.zeros_like(resid,dtype=bool)
        goodp[finitep] = (np.abs(resid[finitep]) < (sigreject * madev))
    else:
        goodp = finitep ## Start with the finite points
        
    for iter in range(iteration):
        if (useSpline == True) & (knots is not None):
            pointsThreshold = len(knots) + polyord
        else:
            pointsThreshold = polyord
        
        if np.sum(goodp) <= pointsThreshold:
            warntext = "Less than "+str(polyord)+"points accepted, returning flat line"
            warnings.warn(warntext)
            
            if useSpline == True:
                spl = UnivariateSpline([0,1,2],[0,0,0],k=1)
            else:
                coeff = np.zeros(polyord + 1)
                coeff[0] = 1.0
        else:
            if useSpline == True:
                
                if knots is None:
                    spl = UnivariateSpline(x[goodp], y[goodp], k=polyord, s=sSpline)
                else:
                    try:
                        spl = LSQUnivariateSpline(x[goodp], y[goodp], knots, k=polyord)
                    except ValueError as inst:
                        knownFailures = ((str(inst) == 'Interior knots t must satisfy Schoenberg-Whitney conditions') | 
                                         ("The input parameters have been rejected by fpchec." in str(inst)))
                        if knownFailures:
                            warnings.warn("Spline fitting failed because of Schoenberg-Whitney conditions. Trying to eliminate knots without sufficient data")
                            
                            if plotEachStep == True:
                                plt.plot(x[goodp],y[goodp],'o',label='data')
                                plt.plot(knots,np.ones_like(knots) * np.median(y[goodp]),'o',label='knots',markersize=10)
                            
                            keepKnots = np.zeros_like(knots,dtype=bool)
                            nKnots = len(knots)
                            for ind,oneKnot in enumerate(knots):
                                if ind == 0:
                                    if np.sum(x[goodp] < oneKnot) > 0:
                                        keepKnots[ind] = True
                                elif ind == nKnots - 1:
                                    if np.sum(x[goodp] > oneKnot) > 0:
                                        keepKnots[ind] = True
                                else:
                                    pointsTest = ((np.sum((x[goodp] > knots[ind-1]) & (x[goodp] < oneKnot)) > 0 ) &
                                                  (np.sum((x[goodp] > oneKnot) & (x[goodp] < knots[ind+1])) > 0 ))
                                    if pointsTest == True:
                                        keepKnots[ind] = True
                            if plotEachStep == True:
                                plt.plot(knots[keepKnots],np.ones_like(knots[keepKnots]) * np.median(y[goodp]),'o',label='knots to keep')
                                plt.show()
                            
                            knots = knots[keepKnots] 
                            spl = LSQUnivariateSpline(x[goodp], y[goodp], knots, k=polyord)
                            
                        else:
                            raise inst
                ymod = spl(x)
            else:
                coeff = np.polyfit(x[goodp],y[goodp],polyord)
                yPoly = np.poly1d(coeff)
                ymod = yPoly(x)
            
            resid = np.abs(ymod - y)
            madev = np.nanmedian(resid)
            if madev > 0:
                ## replacing the old line to avoid runtime errors
                ## goodp = (np.abs(resid) < (sigreject * madev))
                goodp = np.zeros_like(resid,dtype=bool)
                goodp[finitep] = (np.abs(resid[finitep]) < (sigreject * madev))
        
        if plotEachStep == True:
            plt.plot(x,y,'o')
            plt.plot(x[goodp],y[goodp],'o')
            plt.plot(x,ymod)
            plt.show()
    
    if useSpline == True:
        return spl
    else:
        return coeff

def flatten(x,y,flatteningMethod='filter',polyOrd=2,
            highPassFreq=0.01,normalize=True,
            lowPassFreq=None):
    """
    Flatten a time series/array
    """
    
    if flatteningMethod == 'polynomial':
        polyFit = robust_poly(x,y,polyord=polyOrd)
        
        yFlat = y - np.polyval(polyFit,x)
        if normalize == False:
            yFlat = yFlat + np.median(y)
            
    elif flatteningMethod == 'filter':
        
        if lowPassFreq == None:
            sos = signal.butter(5,highPassFreq, 'highpass',analog=False, output='sos')
        else:
            sos = signal.butter(5,[highPassFreq,lowPassFreq], 'bandpass',analog=False, output='sos')
        
        yFlat = signal.sosfiltfilt(sos, y)
        if normalize == False:
            yFlat = yFlat + np.median(y)
    
    return yFlat

def roll_pad(y,pixShift,pad_value=np.nan,order=1):
    """
    Similar as numpy roll (shifting) but make sure the wrap-arounds are NaN
    Converts to floating point since the Nans are weird for integer arrays
    
    If pixShift is an integer, it does whole-pixel shifts
    Otherwise, it will do linear interpolation to do the subpixel shift
    """
    if type(pixShift) is int:
        rolled = np.array(np.roll(y,pixShift),dtype=float)
        if pixShift > 0:
            rolled[0:pixShift] = pad_value
        elif pixShift < 0:
            rolled[pixShift:] = pad_value
        return rolled
    else:
        rolled = ndimage.interpolation.shift(np.array(y,dtype=float),pixShift,
                                             mode='constant',cval=pad_value,
                                             order=order)
        
        return rolled
        # numPix = len(y)
        # indArr = np.arange(numPix)
        # intpixShift = np.int(np.floor(pixShift))
        # rolled = np.zeros_like(y,dtype=float) * np.nan
        # fInterp = interp1d(indArr,y)
        #
        # newFracIndArr = indArr - pixShift
        # pdb.set_trace()
        # if pixShift == 0.0:
        #     rolled = y
        # elif (intpixShift < 0) & (intpixShift >= -(numPix)):
        #     rolled[abs(intpixShift)-1:numPix] = fInterp(newFracIndArr[abs(intpixShift)-1:numPix])
        # elif (intpixShift >= 0) & (intpixShift < numPix):
        #     rolled[0:numPix-intpixShift-1] = fInterp(newFracIndArr[0:numPix-intpixShift-1])
        #
        # return rolled
    


def test_roll(length):
    """
    Test the rolling function
    """
    tmp = [1,2,1,7,1]
    plt.plot(tmp)
    plt.plot(roll_pad(tmp,length))
    plt.show()

def subpixel_peak(x,y):
    polyFit = robust_poly(x,y,polyord=2)
    if len(polyFit) != 3:
        ## Peak fitting failed
        xPeak, yPeak, yModel = np.nan, np.nan, np.nan
    else:
        yModel = np.polyval(polyFit,x)
        xPeak = -polyFit[1]/(2. * polyFit[0])
        yPeak = polyFit[2] - polyFit[1]**2 / (4. * polyFit[0])
    
    return xPeak, yPeak, yModel

def crosscor_offset(x,y1,y2,Noffset=150,diagnostics=False,
                    flatteningMethod='filter',
                    polyOrd=2,
                    highPassFreq=0.01,lowPassFreq=None,
                    subPixel=False):
    """
    Cross correlate two arrays to find the offset
    
    First, a filter is applied to each array to remove low frequency stuff
                    
    Parameters
    -----------
    x: numpy array
        Array for the x values
    
    y1: numpy array
        The first signal assumed to be the reference
    
    y2: numpy array
        The second signal where the shift is desired to the reference
    
    Noffset: int
        number of offset points to explore
    
    diagnostics: bool
        Show diagnostic plots?
                    
    flatteningMethod: str
        What kind of flattening method should be used on the arrays?
            'filter' will apply a filter
            'polynomial' will divide by a polynomial
    
    polyOrd: int
        Order of polynomial to use for flattening,
        passed to :code:`flatten`
    
    highpassFreq: float
        The frequency (on a scale from Nyquist to 1) to pass information
    
    subPixel: bool
        Fit the cross-correlation at the subpixel level?
    """
    
    Npts = len(x)
    
    offsetIndices = (np.arange(2 * Noffset + 1) - Noffset)
    offsets = np.median(np.diff(x)) * offsetIndices
    
    y1Norm = y1 / np.median(y1)
    y2Norm = y2 / np.median(y2)
    
    if diagnostics == True:
        fig, ax = plt.subplots()
        ax.plot(y1Norm)
        tmp = y2Norm[Noffset:Npts-Noffset]
        xTmp = np.arange(len(tmp))
        ax.plot(xTmp + Noffset,tmp)
        plt.show()
    
    y1Flat = flatten(x,y1Norm,flatteningMethod=flatteningMethod,highPassFreq=highPassFreq,
                              lowPassFreq=lowPassFreq,polyOrd=polyOrd)
    y2Flat = flatten(x,y2Norm,flatteningMethod=flatteningMethod,highPassFreq=highPassFreq,
                              lowPassFreq=lowPassFreq,polyOrd=polyOrd)
    
    if diagnostics == True:
        plt.plot(y1Flat,label='reference')
        plt.plot(y2Flat,label='input')
        plt.legend()
        plt.show()
    
    corr = signal.correlate(y1Flat, y2Flat[Noffset:Npts-Noffset], mode='valid')
    
    if subPixel == True:
        indOffset, yPeak, yModel = subpixel_peak(offsetIndices,corr)
        if indOffset < np.min(offsetIndices):
            xOffset = np.min(offsets)
        elif indOffset > np.max(offsetIndices):
            xOffset = np.max(offsets)
        else:
            fInterp = interp1d(offsetIndices,offsets)
            xOffset = float(fInterp(indOffset))
    else:
        peakArg = np.argmax(corr)
        yPeak = corr[peakArg]
        
        xOffset = offsets[peakArg]
        indOffset = offsetIndices[peakArg]

    
    if diagnostics == True:
        plt.plot(offsetIndices,corr,label='Cross-cor')
        if subPixel == True:
            plt.plot(offsetIndices,yModel,label='Parabola Fit')
        plt.plot(indOffset,yPeak,'o',color='red',label='Peak')
        plt.legend()
        plt.show()
    
    if diagnostics == True:
        print("Shift = {}, or index {}".format(xOffset,indOffset))
        plt.plot(y1Flat,label='reference')
        plt.plot(roll_pad(y2Flat,indOffset),label='shifted input')
        plt.legend()
        plt.show()
    
    return xOffset, indOffset

def test_crosscor_offsets():
    x = np.arange(1000)
    y1 = np.random.randn(1000) + 1.
    y2 = y1 + np.random.randn(1000) * 0.1
    
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()
    
    offset, offsetInd = crosscor_offset(x,y1,y2,diagnostics=True)



def get_baseDir():
    if 'TSHIRT_DATA' in os.environ:
        baseDir = os.environ['TSHIRT_DATA']
    else:
        baseDir = os.path.join(os.environ['HOME'],'tshirt_data')
        if os.path.exists(baseDir) == False:
            os.mkdir(baseDir)
    
    return baseDir

def mod_phase_near_zero(x):
    """
    Calculate orbital phases so they are near 0
    """
    return np.mod(np.mod(x,1.0) - 0.5,1.) - 0.5
