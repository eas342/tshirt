#import phot_pipeline
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy import signal
from phot_pipeline import robust_poly
import matplotlib.pyplot as plt



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
            highPassFreq=0.01,normalize=True):
    """
    Flatten a time series/array
    """
    
    if flatteningMethod == 'polynomial':
        polyFit = robust_poly(x,y,polyOrd=polyOrd)
        
        yFlat = y / np.polyval(polyFit,x)
        if normalize == False:
            yFlat = yFlat + np.median(y)
            
    elif flatteningMethod == 'filter':
        
        #sos = signal.butter(3,[1e-6,0.1], 'bandpass',analog=False, output='sos')
        sos = signal.butter(5,highPassFreq, 'highpass',analog=False, output='sos')
        
        yFlat = signal.sosfiltfilt(sos, y)
        if normalize == False:
            yFlat = yFlat + np.median(y)
    
    return yFlat

def crosscor_offset(x,y1,y2,Noffset=150,diagnostics=False,
                    flatteningMethod='filter',
                    highPassFreq=0.01):
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
    
    """
    
    Npts = len(x)
    Noffset = 150
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
    
    y1Flat = flatten(x,y1Norm,flatteningMethod=flatteningMethod,highPassFreq=highPassFreq)
    y2Flat = flatten(x,y2Norm,flatteningMethod=flatteningMethod,highPassFreq=highPassFreq)
    
    if diagnostics == True:
        plt.close()
        plt.plot(y1Flat)
        plt.plot(y2Flat)
        plt.show()
    
    corr = signal.correlate(y1Flat, y2Flat[Noffset:Npts-Noffset], mode='valid')
    
    peakArg = np.argmax(corr)
    
    if diagnostics == True:
        plt.plot(offsets,corr)
        plt.plot(offsets[peakArg],corr[peakArg],'o',color='red')
        plt.show()
    
    if diagnostics == True:
        print("Shift = {}, or index {}".format(offsets[peakArg],offsetIndices[peakArg]))
        plt.plot(y1Flat)
        plt.plot(np.roll(y2Flat,offsetIndices[peakArg]))
        plt.show()
        
    return offsets[peakArg], offsetIndices[peakArg]

def test_crosscor_offsets():
    x = np.arange(1000)
    y1 = np.random.randn(1000) + 1.
    y2 = y1 + np.random.randn(1000) * 0.1
    
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()
    
    offset, offsetInd = crosscor_offset(x,y1,y2,diagnostics=True)
    