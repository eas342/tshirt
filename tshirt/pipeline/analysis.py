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
from .utils import robust_poly
from .phot_pipeline import phot
from .spec_pipeline import spec
from copy import deepcopy
import os
import tqdm

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
        
        if phot_obj.param['scaleAperture'] == True:
            param['apRadius'] = 1.0
            param['apScale'] = srcSize
        else:
            param['apRadius'] = srcSize
        param['backStart'] = backStart
        param['backEnd'] = backEnd
        
        
        new_phot = phot(directParam=param)
        
        if showPlot == True:
            new_phot.showStamps(boxsize=plotHeight)
        
    elif phot_obj.pipeType == 'spectroscopy':
        param['apWidth'] = srcSize
        if param['traceCurvedSpectrum'] == False:
            backLoc_1_start = param['starPositions'][0] - backEnd
            backLoc_1_end = param['starPositions'][0] - backStart
            backLoc_2_start = param['starPositions'][0] + backStart
            backLoc_2_end = param['starPositions'][0] + backEnd
            backLocList = [[backLoc_1_start,backLoc_1_end],
                           [backLoc_2_start,backLoc_2_end]]
            if param['dispDirection'] == 'x':
                param['bkgRegionsY'] = backLocList
            else:
                param['bkgRegionsX'] = backLocList
        else:
            param['backgMinRadius'] = backStart
        
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
        
        if phot_obj.pipeType == 'photometry':
            new_phot.do_phot(useMultiprocessing=True)
            noiseTable = new_phot.print_phot_statistics(refCorrect=True,returnOnly=True,shorten=shorten)
            
        elif phot_obj.pipeType == 'spectroscopy':
            new_phot.do_extraction(useMultiprocessing=True)
            noiseTable = new_phot.print_noise_wavebin(shorten=shorten,nbins=1,recalculate=True)
        
        else:
            raise Exception("Unrecognized pipeType")
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
    if phot_obj.pipeType == 'photometry':
        tableDir = 'phot_aperture_optimization'
    elif phot_obj.pipeType == 'spectroscopy':
        tableDir = 'spec_aperture_optimization'
    else:
        raise Exception("Unrecognized pipeType")
    
    outTable_path = os.path.join(new_phot.baseDir,'tser_data',tableDir,outTable_name)
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

def backsub_list(specObj,outDirectory='tmp'):
    """
    Save all the background-subtracted images
    
    Parameters
    -----------
    specObj: a `any::tshirt.pipeline.spec_pipeline.spec` object
    """
    
    for i in tqdm.tqdm(range(len(specObj.fileL))):
        oneFile = specObj.fileL[i]
        # read in
        
        img, head = specObj.getImg(oneFile)
        
        # do backsub
        subImg, bkgModel, subHead = specObj.do_backsub(img,head,i,saveFits=False,
                                                       directions=specObj.param['bkgSubDirections'])
        
        ## save
        baseName = os.path.splitext(os.path.basename(oneFile))[0]
        outName = baseName + '_backsub.fits'
        outPath = os.path.join(outDirectory,outName)
        HDUList = fits.PrimaryHDU(subImg,subHead)
        HDUList.writeto(outPath)
    
    
