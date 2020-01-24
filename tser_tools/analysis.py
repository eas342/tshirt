#import phot_pipeline
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
from phot_pipeline import robust_poly
import matplotlib.pyplot as plt
import pdb

def bin_examine():
    t = Table.read('tser_data/refcor_phot/refcor_phot_S02illum1miAll_GLrun104.fits')
    
    allStd, allBinSize = [], []

    binsTry = [500,100,50,30,10]
    for oneBin in binsTry:
        yBin, xEdges, binNum = binned_statistic(t['Time'],t['Y Corrected'],
                                                statistic='mean',bins=oneBin)
        stdBinned = np.std(yBin)
        binSize = (xEdges[1] - xEdges[0]) * 24. * 60.
        allStd.append(stdBinned)
        allBinSize.append(binSize)

    print("{} bin ({} min) std = {} ppm".format(binsTry[-1],allBinSize[-1],allStd[-1]))
    plt.loglog(allBinSize,allStd)
    plt.show()
    print

def flatten(x,y,flatteningMethod='filter',polyOrd=2,
            highPassFreq=0.01,normalize=True,
            lowPassFreq=None):
    """
    Flatten a time series/array
    """
    
    if flatteningMethod == 'polynomial':
        polyFit = robust_poly(x,y,polyOrd=polyOrd)
        
        yFlat = y / np.polyval(polyFit,x)
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
        rolled = np.array(np.roll(y,pixShift),dtype=np.float)
        if pixShift > 0:
            rolled[0:pixShift] = pad_value
        elif pixShift < 0:
            rolled[pixShift:] = pad_value
        return rolled
    else:
        rolled = ndimage.interpolation.shift(np.array(y,dtype=np.float),pixShift,
                                             mode='constant',cval=pad_value,
                                             order=order)
        
        return rolled
        # numPix = len(y)
        # indArr = np.arange(numPix)
        # intpixShift = np.int(np.floor(pixShift))
        # rolled = np.zeros_like(y,dtype=np.float) * np.nan
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
                              lowPassFreq=lowPassFreq)
    y2Flat = flatten(x,y2Norm,flatteningMethod=flatteningMethod,highPassFreq=highPassFreq,
                              lowPassFreq=lowPassFreq)
    
    if diagnostics == True:
        plt.plot(y1Flat,label='reference')
        plt.plot(y2Flat,label='input')
        plt.legend()
        plt.show()
    
    corr = signal.correlate(y1Flat, y2Flat[Noffset:Npts-Noffset], mode='valid')
    
    if subPixel == True:
        indOffset, yPeak, yModel = subpixel_peak(offsetIndices,corr)
        fInterp = interp1d(offsetIndices,offsets)
        xOffset = np.float(fInterp(indOffset))
    else:
        peakArg = np.argmax(corr)
        yPeak = corr[peakArg]
        xOffset = offsets[peakArg]
        indOffset = offsetIndices[peakArg]

    
    if diagnostics == True:
        plt.plot(offsetIndices,corr,label='Cross-cor')
        if subPixel == True:
            plt.plot(offsetIndices,yModel,label='Parabola Fit')
        plt.plot(xOffset,yPeak,'o',color='red',label='Peak')
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
    