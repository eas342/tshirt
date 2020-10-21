#import phot_pipeline
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
import matplotlib.pyplot as plt
import pdb
from .phot_pipeline import robust_poly
from .phot_pipeline import phot
from .spec_pipeline import spec
from copy import deepcopy

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
        if indOffset < np.min(offsetIndices):
            xOffset = np.min(offsets)
        elif indOffset > np.max(offsetIndices):
            xOffset = np.max(offsets)
        else:
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


def adjust_aperture_set(phot_obj,param,srcSize,backStart,backEnd,
                        showPlot=False,plotHeight=None):
    """
    Takes a photometry or spectroscopy object and sets the aperture parameters
    
    Parameters
    ------------
    phot_obj: a tshirt phot or spec object
        Input object to be adjusted
    
    param: dictionary
        A parameter dictionary for a phot or spec object
    
    showPlot: bool
        Show a plot of the aperture set?
                        
    plotHeight: float
        A size of the photometry plot (only works for photometry)
    
    
    Returns
    --------
    new_phot: a tshirt phot or spec object
        Spec or phot object w/ modified parameters
    """
    
    if phot_obj.pipeType == 'photometry':
        param['apRadius'] = srcSize
        param['backStart'] = backStart
        param['backEnd'] = backEnd
        
        new_phot = phot(directParam=param)
        
        if showPlot == True:
            new_phot.showStamps(boxsize=plotHeight)
        
    elif phot_obj.pipeType == 'spectroscopy':
        param['apWidth'] = srcSize
        backLoc_1_start = param['starPositions'][0] - backEnd
        backLoc_1_end = param['starPositions'][0] - backStart
        backLoc_2_start = param['starPositions'][0] + backStart
        backLoc_2_end = param['starPositions'][0] + backEnd
        backLocList = [[backLoc_1_start,backLoc_1_end],
                       [backLoc_2_start,backLoc_2_end]]
        if param['dispDirection'] == 'x':
            param['bkgRegionsX'] = backLocList
        else:
            param['bkgRegionsY'] = backLocList
        
        new_phot = spec(directParam=param)
        if showPlot == True:
            new_phot.showStarChoices()
    else:
        raise Exception("Unrecognized pipeType")
    
    return new_phot

def aperture_size_sweep(phot_obj,stepSize=5,srcRange=[5,20],backRange=[5,28],
                        minBackground=2,stepSizeSrc=None,stepSizeBack=None,
                        shorten=False):
    """
    Calculate the Noise Statistics for a "Sweep" of Aperture Sizes
    Loops through a series of source sizes and background sizes in a grid search
    
    Parameters
    -----------
    stepSize: float
        The step size. It will be superseded by stepSize_bck or stepSizeSrc if used.
    srcRange: two element list
        The minimum and maximum src radii to explore
    backRange: two element list
        The minimum and maximum aperture radii to explore (both for the inner & outer)
    minBackground: float
        The minimum thickness for the background annulus or rectangle width
    stepSizeSrc: float
        (optional) Specify the step size for the source that will supersed the general
        stepSize
    stepSizeBack: float
        (optional) Specify the step size for the background that will supersed the general
        stepSize
    shorten: bool
        Shorten the time series? (This is passed to print_phot_statistics)
    """
    
    if stepSizeSrc == None:
        stepSizeSrc = stepSize
    if stepSizeBack == None:
        stepSizeBack = stepSize
    
    ## change the short name to avoid over-writing previous files
    origParam = deepcopy(phot_obj.param)
    origName = origParam['srcNameShort']
    param = deepcopy(origParam)
    
    ## show the most compact and most expanded configurations
    srcPlots = srcRange
    backStartPlots = [np.max([srcRange[0],backRange[0]]),
                      backRange[1] - minBackground]
    backEndPlots = [backStartPlots[0] + minBackground,
                    backRange[1]]
    plotHeights = [backRange[1]+5,backRange[1] + 5]
    for i, oneConfig in enumerate(['compact','expanded']):
        param['srcNameShort'] = origName + '_' + oneConfig
        new_phot = adjust_aperture_set(phot_obj,param,srcRange[i],
                                       backStartPlots[i],backEndPlots[i],
                                       showPlot=True,plotHeight=plotHeights[i])
    
    apertureSets = []
    t = Table(names=['src','back_st','back_end'])
    for srcSize in np.arange(srcRange[0],srcRange[1],stepSizeSrc):
        ## start from the backRange min or the src, whichever is bigger
        back_st_minimum = np.max([srcSize,backRange[0]])
        
        ## finish at the backRange max, but allow thickness
        back_st_maximum = backRange[1] - minBackground
        
        for back_st in np.arange(back_st_minimum,back_st_maximum,stepSizeBack):
            ## start the outer background annulus, at least minBackground away
            back_end_minimum = back_st + minBackground
            back_end_maximum = backRange[1]
            for back_end in np.arange(back_end_minimum,back_end_maximum,stepSize):
                apertureSets.append([srcSize,back_st,back_end])
                t.add_row([srcSize,back_st,back_end])
                #print("src: {}, back st: {}, back end: {}".format(srcSize,back_st,back_end))
                
    ## for the rest, it will save a common file name
    param['srcNameShort'] = origName + '_aperture_sizing'
    
    stdevArr, theo_err, mad_arr = [], [], []
    for i,apSet in enumerate(apertureSets):
        print("src: {}, back st: {}, back end: {}".format(apSet[0],apSet[1],apSet[2]))
        new_phot = adjust_aperture_set(phot_obj,param,apSet[0],apSet[1],apSet[2],
                                       showPlot=False)
        
        new_phot.do_phot(useMultiprocessing=True)
        noiseTable = new_phot.print_phot_statistics(refCorrect=True,returnOnly=True,shorten=shorten)
        stdevArr.append(noiseTable['Stdev (%)'][0])
        theo_err.append(noiseTable['Theo Err (%)'][0])
        mad_arr.append(noiseTable['MAD (%)'][0])
    t['stdev'] = stdevArr
    t['theo_err'] = theo_err
    t['mad_arr'] = mad_arr
    
    outTable_name = 'aperture_opt_{}_src_{}_{}_step_{}_back_{}_{}_step_{}.csv'.format(new_phot.dataFileDescrip,
                                                                              srcRange[0],srcRange[1],stepSizeSrc,
                                                                              backRange[0],backRange[1],
                                                                              stepSizeBack)
    outTable_path = os.path.join(new_phot.baseDir,'tser_data','phot_aperture_optimization',outTable_name)
    t.write(outTable_path,overwrite=True)
    
    print('Writing table to {}'.format(outTable_path))
    
    ind = np.argmin(t['stdev'])
    print("Min Stdev results:")
    
    print(t[ind])
    
    return t

def plot_apsizes(apertureSweepFile,showPlot=True):
    """
    Plot the aperture sizes calculated from :any:`aperture_size_sweep`
    
    Parameters
    ----------
    apertureSweepFile: str
        A .csv file created by aperture_size_sweep
    showPlot: bool
        Show the plot w/ matplotlib? otherwise, it saves to file
    """
    
    dat = ascii.read(apertureSweepFile)
    
    fig, axArr2D = plt.subplots(3,3,sharey=True)
    
    ## share axes along columns
    for oneColumn in [0,1,2]:
        axTop = axArr2D[0,oneColumn]
        axMid = axArr2D[1,oneColumn]
        axBot = axArr2D[2,oneColumn]
        axTop.get_shared_x_axes().join(axMid, axBot)
    
    labels = ['Source Radius','Back Start','Back End']
    keys = ['src','back_st','back_end']
    
    statistics = ['stdev','theo_err','mad_arr']
    for statInd,statistic in enumerate(statistics):
        axArr1D = axArr2D[statInd]
        for ind, ax in enumerate(axArr1D):
            ax.semilogy(dat[keys[ind]],dat[statistic],'.')
            ax.set_xlabel(labels[ind])
            if ind==0:
                ax.set_ylabel(statistic)
    if showPlot == True:
        fig.show()
    else:
        raise NotImplementedError
        