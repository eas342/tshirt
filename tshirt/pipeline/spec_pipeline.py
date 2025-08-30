import photutils
from astropy.io import fits, ascii
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pkg_resources
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
import pdb
from copy import deepcopy
import yaml
import warnings
from scipy.stats import binned_statistic
from scipy.stats import norm
from scipy.interpolate import interp1d
import astropy
from astropy.table import Table
try:
    from astropy.stats import LombScargle
except ImportError as errLS:
    from astropy.timeseries import LombScargle
import multiprocessing
from multiprocessing import Pool
import tqdm
try:
    import bokeh.plotting
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.models import Range1d
    from bokeh.models import WheelZoomTool
    from bokeh.palettes import Dark2_5 as palette
    # itertools handles the cycling
    import itertools
except ImportError as err2:
    print("Could not import bokeh plotting. Interactive plotting may not work")

try:
    from astropy.modeling import models, fitting
except ImportError as err3:
    print("Could not import astropy modeling for spatial profiles")

from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from . import phot_pipeline
from . import utils
from . import instrument_specific

path_to_example = "parameters/spec_params/example_spec_parameters.yaml"
exampleParamPath = pkg_resources.resource_filename('tshirt',path_to_example)

path_to_defaults = "parameters/spec_params/default_params.yaml"
defaultParamPath = pkg_resources.resource_filename('tshirt',path_to_defaults)

import traceback
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

#warnings.showwarning = warn_with_traceback

class spec(phot_pipeline.phot):
    def __init__(self,paramFile=exampleParamPath,
                 directParam=None):
        """ Spectroscopy class
    
        Parameters
        ------
        paramFile: str
            Location of the YAML file that contains the photometry parameters as long
            as directParam is None. Otherwise, it uses directParam
        
        Properties
        -------
        paramFile: str
            Same as paramFile above
        param: dict
            The photometry parameters like file names, aperture sizes, guess locations
        fileL: list
            The files on which photometry will be performed
        nImg: int
            Number of images in the sequence
        directParam: dict
            Parameter dictionary rather than YAML file (useful for batch processing)
        """
        self.pipeType = 'spectroscopy'
        self.get_parameters(paramFile=paramFile,directParam=directParam)
        
        defaultParams = phot_pipeline.read_yaml(defaultParamPath)
        
        for oneKey in defaultParams.keys():
            if oneKey not in self.param:
                self.param[oneKey] = defaultParams[oneKey]
        
        # Get the file list
        self.check_file_structure()
        self.photFile = 'none'
        self.fileL = self.get_fileList()
        self.nImg = len(self.fileL)
        
        self.nsrc = len(self.param['starPositions'])
        
        self.srcNames = np.array(np.arange(self.nsrc),dtype=str)
        self.srcNames[0] = 'src'
        
        ## Set up file names for output
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        specFile_name = 'spec_'+self.dataFileDescrip+'.fits'
        self.specFile = os.path.join(self.baseDir,'tser_data','spec',specFile_name)
        
        dyn_specFileName_prefix = 'dyn_spec_{}'.format(self.dataFileDescrip)
        self.dyn_specFile_prefix = os.path.join(self.baseDir,'tser_data','dynamic_spec',
                                                dyn_specFileName_prefix)
        
        wavebin_fileName_prefix = 'wavebin_spec_{}'.format(self.dataFileDescrip)
        self.wavebin_file_prefix = os.path.join(self.baseDir,'tser_data','wavebin_spec',
                                                wavebin_fileName_prefix)
        
        self.profile_dir = os.path.join(self.baseDir,'tser_data','saved_profiles')
        self.weight_dir = os.path.join(self.baseDir,'tser_data','saved_weights')

        self.traceReference = self.param['traceReference']
        if self.traceReference is None:
            self.traceFile = os.path.join(self.baseDir,'traces','trace_functions',
                                        'trace_'+self.dataFileDescrip+'.ecsv')
            self.traceDataFile = os.path.join(self.baseDir,'traces','trace_data',
                                            'trace_data_'+self.dataFileDescrip+'.ecsv')
        else:
            self.traceDataFile = None
            self.traceFile = self.traceReference

        self.peakSpecFile = os.path.join(self.baseDir,'tser_data','peak_spec',
                                         'peak_spec_{}.ecsv'.format(self.dataFileDescrip))

        self.check_trace_requirements()

        self.master_profile_prefix = 'master_{}'.format(self.dataFileDescrip)
        #self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'
        #self.refCorPhotFile = 'tser_data/refcor_phot/refcor_'+self.dataFileDescrip+'.fits'
        self.get_summation_direction()
        
        ## a little delta to add to add to profile so that you don't get log(negative)
        if self.param['splineFloor'] is None:
            self.floor_delta = self.param['readNoise'] * 2. 
        else:
            self.floor_delta = self.param['splineFloor']
        
        ## minimum number of pixels to do 
        self.minPixForCovarianceWeights = 3
        
        ## set up the dispersion offsets (if any)
        self.set_up_disp_offsets()
        
        self.check_parameters()
        
        ## set up file name and path for flux cal results
        self.fluxCal_name = 'fluxcal_{}.ecsv'.format(self.dataFileDescrip)
        
        self.fluxCal_path = os.path.join(self.baseDir,'tser_data',
                                         'fluxcal_spec',self.fluxCal_name)

        ## set up file name and path for flux cal results
        self.fluxCal2D_name = 'fluxcal_2D_{}.fits'.format(self.dataFileDescrip)

        self.fluxCal2D_path = os.path.join(self.baseDir,'tser_data',
                                            'fluxcal2D_spec',self.fluxCal2D_name)
        
        
    def check_parameters(self):
        if ('bkgSubY' in self.param) | ('bkgSubY' in self.param):
            bkgSubDirections = []
            if ('bkgSubY' in self.param):
                if self.param['bkgSubY'] == True:
                    bkgSubDirections.append('Y')
            else:
                bkgSubDirections.append('Y')
            if ('bkgSubX' in self.param):
                if self.param['bkgSubX'] == True:
                    bkgSubDirections.append('X')
            else:
                bkgSubDirections.append('X')
                    
            warnings.warn('Deprecated parameter bkgSubY used in parameter file. Setting bkgSubDirections to [{}]'.format(' '.join(bkgSubDirections)))
            self.param['bkgSubDirections'] = bkgSubDirections
            
        
        dispCheck = (self.param['dispDirection'] == 'x') | (self.param['dispDirection'] == 'y')
        assert dispCheck, 'Dispersion direction parameter not valid'
        if self.param['readNoiseCorrelation'] == True:
            assertText = 'Ap width not big enough to use read noise covariance estimates'
            assert (self.param['apWidth'] > self.minPixForCovarianceWeights),assertText 
        
        if self.param['dispOffsets'] is not None:
            assert len(self.param['dispOffsets']) == self.nsrc,'Dispersion offsets needs to match number of sources'
    
        if self.param['mosBacksub'] == True:
            assertText = 'MOS Backsub currently only allows one subtraction direction'
            assert len(self.param['bkgSubDirections']) <= 1,assertText
            if len(self.param['bkgSubDirections']) == 1:
                assertText = 'MOS Backsub currently only allows cross-dispersion subtraction'
                assert self.param['dispDirection'].lower() != self.param['bkgSubDirections'][0].lower(),assertText
    
    def set_up_disp_offsets(self):
        if self.param['dispOffsets'] is None:
            self.dispOffsets = np.zeros(self.nsrc)
        else:
            self.dispOffsets = self.param['dispOffsets']
    
    def get_summation_direction(self):
        if self.param['dispDirection'] == 'x':
            self.spatialAx = 0 ## summation axis along Y (spatial axis)
            self.dispAx = 1 ## dispersion axis is X
        else:
            self.spatialAx = 1 ## summation axis along X (spatial axis)
            self.dispAx = 0 ## dispersion axis is 0
    
    def add_parameters_to_header(self,header=None):
        if header is None:
            header = fits.Header()
        
        ## max depth to dig in lists of lists of lists...
        maxDepth = 3
        keyList = np.sort(list(self.param.keys()))
        for oneKey in keyList:
            if len(oneKey) > 8:
                keyName = oneKey[0:8]
            else:
                keyName = oneKey
            
            metaDatum = self.param[oneKey]
            if type(metaDatum) == list:
                for ind1, item1 in enumerate(metaDatum):
                    if type(item1) == list:
                        for ind2,item2 in enumerate(item1):
                            if type(item2) == list:
                                warnings.warn("3-Deep lists not saved to output FITS header")
                            else:
                                if len(keyName) > 6:
                                    keyName = keyName[0:6]
                                useKey = "{}{}{}".format(keyName,ind1,ind2)
                                header[useKey] = (item2, "{}: item: {} sub-item {}".format(oneKey,ind1,ind2))
                            
                    else:
                        if len(keyName) > 7:
                            keyName = keyName[0:7]
                        
                        useKey = "{}{}".format(keyName,ind1)
                        header[useKey] = (item1, "{}: item {}".format(oneKey,ind1))
            else:
                header[keyName] = (metaDatum, oneKey)
                
                # if value != None:
                #     ## dig as far as the depth number to check for a list
                #     for oneDepth in np.arange(depth):
                #         value = value[0]
                # if type(value) == list:
                #     self.paramLists.append(oneKey)
                #     self.counts.append(len(self.batchParam[oneKey]))
        return header
    
    def get_median_img(self,nImg):
        """
        Calculate the median image fron n images

        Parameters
        ----------
        nImg: int
            Number of images to use for the profile
        """
        nSpacing = self.nImg // nImg
        indToUse = np.arange(0,self.nImg,nSpacing)
        nUsed = len(indToUse)
        img, head = self.get_default_im()
        all_img = np.zeros([nUsed,img.shape[0],img.shape[1]])
        
        for count,oneInd in enumerate(indToUse):
            oneImgName = self.fileL[oneInd]
            img2, head2 = self.getImg(oneImgName)
            all_img[count,:,:] = img2
        medImg = np.median(all_img,axis=0)
        return medImg, head


    def do_extraction(self,useMultiprocessing=False):
        """
        Extract all spectroscopy
        """
        from ..__init__ import __version__
        
        fileCountArray = np.arange(self.nImg)
        
        if self.param['fixedProfile'] == True:
            if (self.nImg > self.param['nImgForProfile']) & (self.param['nImgForProfile'] > 1):
                img, head = self.get_median_img(self.param['nImgForProfile'])
            else:
                img, head = self.get_default_im()
            
            imgSub, bkgModel, subHead = self.do_backsub(img,head,saveFits=False,
                                                        directions=self.param['bkgSubDirections'])
            profileList, smooth_img_list = self.find_profile(imgSub,subHead,saveFits=True,masterProfile=True)
            
            if (self.param['readNoiseCorrelation'] == True):
                ## Only run the read noise inverse covariance matrix scheme once for fixed profile
                
                readNoise = self.get_read_noise(head)
                ## Background and read noise only.
                ## Smoothed source flux added below
                varImg = readNoise**2 + bkgModel ## in electrons because it should be gain-corrected
                
                for oneSrc in np.arange(self.nsrc): ## use the smoothed flux for the variance estimate
                    varImg = varImg + np.abs(smooth_img_list[oneSrc]) ## negative flux is approximated as photon noise
            
                dispAx = self.dispAx
                ## dispersion indices in pixels (before wavelength calibration)
                nDisp = img.shape[dispAx]
                
                for oneSrc in np.arange(self.nsrc):
                    profile_img = profileList[oneSrc]
                    
                    
                    self.find_cov_weights(nDisp,varImg,profile_img,readNoise,
                                          src=oneSrc,saveWeights=True,diagnoseCovariance=False)
        
        if useMultiprocessing == True:
            outputSpec = phot_pipeline.run_multiprocessing_phot(self,fileCountArray,method='spec_for_one_file')
        else:
            outputSpec = []
            for ind in tqdm.tqdm(fileCountArray):
                outputSpec.append(self.spec_for_one_file(ind))
        
        timeArr = []
        airmass = []
        dispPixelArr = outputSpec[0]['disp indices']
        nDisp = len(dispPixelArr)
        optSpec = np.zeros([self.nsrc,self.nImg,nDisp])
        optSpec_err = np.zeros_like(optSpec)
        sumSpec = np.zeros_like(optSpec)
        sumSpec_err = np.zeros_like(optSpec)
        backSpec = np.zeros_like(optSpec)
        
        refRows = np.zeros([self.nImg,nDisp]) * np.nan
        
        cenArr = np.zeros([self.nsrc,self.nImg]) * np.nan
        fwhmArr = np.zeros([self.nsrc,self.nImg]) * np.nan
        
        for ind in fileCountArray:
            specDict = outputSpec[ind]
            timeArr.append(specDict['t0'].jd)
            optSpec[:,ind,:] = specDict['opt spec']
            optSpec_err[:,ind,:] = specDict['opt spec err']
            sumSpec[:,ind,:] = specDict['sum spec']
            sumSpec_err[:,ind,:] = specDict['sum spec err']
            backSpec[:,ind,:] = specDict['back spec']
            if 'ref row' in specDict:
                refRows[ind,:] = specDict['ref row']
            airmass.append(specDict['airmass'])
            
            if 'cen' in specDict:
                cenArr[:,ind] = specDict['cen']
            if 'fwhm' in specDict:
                fwhmArr[:,ind] = specDict['fwhm']
                
        
        hdu = fits.PrimaryHDU(optSpec)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with spectroscopy')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('disp','dispersion axis')
        hdu.header['AXIS2'] = ('image','image axis')
        hdu.header['AXIS3'] = ('src','source axis')
        hdu.header['SRCNAME'] = (self.param['srcName'], 'Source name')
        hdu.header['NIGHT'] = (self.param['nightName'], 'Night Name')
        hdu.header['TSHIRTV'] = (__version__,'tshirt version number')
        hdu.name = 'Optimal Spec'
        
        hdu.header = self.add_parameters_to_header(hdu.header)
        
        hduOptErr = fits.ImageHDU(optSpec_err,hdu.header)
        hduOptErr.name = 'Opt Spec Err'
        
        hduSum = fits.ImageHDU(sumSpec,hdu.header)
        hduSum.name = 'Sum Spec'
        
        hduSumErr = fits.ImageHDU(sumSpec_err,hdu.header)
        hduSumErr.name = 'Sum Spec Err'
        
        hduBack = fits.ImageHDU(backSpec,hdu.header)
        hduBack.name = 'Background Spec'
        
        hduDispIndices = fits.ImageHDU(dispPixelArr)
        hduDispIndices.header['AXIS1'] = ('disp index', 'dispersion index (pixels)')
        hduDispIndices.name = 'Disp Indices'
        
        hduFileNames = self.make_filename_hdu(airmass=airmass)
        
        ## Get an example original header
        exImg, exHeader = self.get_default_im()
        hduOrigHeader = fits.ImageHDU(None,exHeader)
        hduOrigHeader.name = 'Orig Header'
        
        ## Save the times
        hduTime = fits.ImageHDU(np.array(timeArr))
        hduTime.header['AXIS1'] = ('time', 'time in Julian Day (JD)')
        hduTime.name = 'TIME'
        
        ## Save the mean refence pixel rows
        hduRef = fits.ImageHDU(refRows)
        hduRef.header['AXIS1'] = ('X','Image X axis')
        hduRef.header['AXIS2'] = ('image', 'image axis, ie. which integration')
        hduRef.header['BUNIT'] = ('counts', 'should be e- if gain has been applied')
        hduRef.header['VAL'] = ('Mean','Each pixel is the mean of the 4 bottom reference pixels')
        hduRef.name = 'REFPIX'
        
        ## Save the wavelength array
        hduWave = fits.ImageHDU(self.wavecal(dispPixelArr,head=exHeader))
        hduWave.header['AXIS1'] = ('wavelength','wavelength (microns)')
        hduWave.header['BUNIT'] = ('microns','wavelength unit')
        hduWave.name = 'WAVELENGTH'
        
        ## Save the spatial centroid
        hduCen = fits.ImageHDU(cenArr)
        hduCen.header['AXIS1'] = ('image','spatial / time axis')
        hduCen.header['AXIS2'] = ('src','source axis for multiple sources')
        hduCen.header['BUNITS'] = ('px', 'pixels in spatial direction')
        hduCen.name = 'CENTROID'
        
        hduFWHM = fits.ImageHDU(fwhmArr)
        hduFWHM.header['AXIS1'] = ('image','spatial / time axis')
        hduFWHM.header['AXIS2'] = ('src','source axis for multiple sources')
        hduFWHM.header['BUNITS'] = ('px', 'pixels in spatial direction')
        hduFWHM.name = 'FWHM'
        
        HDUList = fits.HDUList([hdu,hduOptErr,hduSum,hduSumErr,
                                hduBack,hduDispIndices,
                                hduTime,hduFileNames,hduOrigHeader,
                                hduRef,hduWave,hduCen,hduFWHM])
        HDUList.writeto(self.specFile,overwrite=True)
        
    
    def backsub_oneDir(self,img,head,oneDirection,saveFits=False,
                       showEach=False,ind=None,custPrefix=None):
        """
        Do the background subtraction in a specified direction
        Either row-by-row or column-by-column
        
        Parameters
        -----------
        img: np.array
            2D image to be subtracted
        head: astropy fits header
            Header of file
        oneDirection: str
            'X' does row-by-row subtraction
            'Y' does column-by-column subtraction
        saveFits: bool
            Save a fits file of the subtraction model?
        showEach: bool
            Show each polynomial fit?
        ind: int or None
            The ind in the fileL. This is mainly used for naming files,
            if files are being saved.
        custPrefix: str or None
            A custom prefix for saved file names of the subtraction.
            If None and ind is None, it saves the prefix as unnamed
            If None and ind is an int, it uses the original file name
        
        Returns
        --------
        imgSub, model, head: tuple of (numpy array, numpy array, :code:`astropy.fits.header`)
            'imgSub' is a background-subtracted image
            'model' is a background model image
            'head' is a header for the background-subtracted image
        
        Example
        --------
        .. code-block:: python
        
            from tshirt.pipeline import spec_pipeline
            spec_pipeline.spec()
            img, head = spec.get_default_im()
            img2, bkmodel2, head2 = spec.backsub_oneDir(img,head,'X')
        
        """

        pts_threshold = 3 ## number of points required for a background subtraction
        ## otherwise, it returns 0
        
        if oneDirection == 'X':
            subtractionIndexArrayLength = img.shape[1]
            cross_subtractionIndexArrayLength = img.shape[0]
        elif oneDirection == 'Y':
            
            subtractionIndexArrayLength = img.shape[0]
            cross_subtractionIndexArrayLength = img.shape[1]
        else:
            raise Exception("Unrecognized subtraction direction")
        
        if self.param['mosBacksub'] == True:
            n_subtractions = self.nsrc ## separate backsub for each source
        else:
            n_subtractions = 1 ## one backsub for the whole image
        
        fitOrder = self.param['bkgOrder{}'.format(oneDirection)]
        ## make a background model
        bkgModel = np.zeros_like(img)
        
        if self.param['traceCurvedSpectrum'] == True:
            apTables = []
            for oneSrc in np.arange(self.nsrc):
                apTables.append(self.calculate_apertures(src=oneSrc))

        ## loop through the number of subtractions
        for sub_counter in np.arange(n_subtractions):
            
            if self.param['mosBacksub'] == True:
                srcInd = sub_counter ## source index
                ## Here it's assumed that the subtraction is in the cross-dispersion direction
                cross_subtractionIndexArray = np.arange(self.param['dispPixels'][0],
                                                        self.param['dispPixels'][1])
                                                        
                cross_subtractionIndexArray = cross_subtractionIndexArray + self.param['dispOffsets'][srcInd]
                
                
                subtractionIndexArray = np.arange(subtractionIndexArrayLength)
                
                ## only subtract points near the source
                absoluteRegions = (np.array(self.param['bkgRegions{}'.format(oneDirection)]) + 
                                   self.param['starPositions'][srcInd])
                lowestPoint = np.min(absoluteRegions)
                highestPoint = np.max(absoluteRegions)
                
                ptsToSubtract = ((subtractionIndexArray >= lowestPoint) & 
                                    (subtractionIndexArray <= highestPoint))
                
            else:
                cross_subtractionIndexArray = np.arange(cross_subtractionIndexArrayLength)
                subtractionIndexArray = np.arange(subtractionIndexArrayLength)
                ## subtract the whole row/column
                ptsToSubtract = np.ones(len(subtractionIndexArray),dtype=bool)
            
            ## set up which points to do background fitting for
            ptsAll = np.zeros(len(subtractionIndexArray),dtype=bool)
            for oneRegion in self.param['bkgRegions{}'.format(oneDirection)]:
                if self.param['mosBacksub'] == True:
                    startSub = int(self.param['starPositions'][srcInd] + oneRegion[0])
                    endSub = int(self.param['starPositions'][srcInd] + oneRegion[1])
                else:
                    startSub, endSub = int(oneRegion[0]),int(oneRegion[1])
                
                ptsAll[startSub:endSub] = True
            
            
            for cross_Ind in cross_subtractionIndexArray:
                ind_var = subtractionIndexArray ## independent variable
                if oneDirection == 'X':
                    dep_var = img[cross_Ind,:]
                else:
                    dep_var = img[:,cross_Ind]

                if self.param['traceCurvedSpectrum'] == True:
                    pts = deepcopy(ptsAll)
                    for oneSrc in np.arange(self.nsrc):
                        oneTable = apTables[oneSrc]
                        disp_pts_match = (cross_Ind == oneTable['dispersion_px'])
                        if np.sum(disp_pts_match) > 0:
                            spatialStart = oneTable['bkgEnd 0'][disp_pts_match][0]
                            spatialEnd = oneTable['bkgStart 1'][disp_pts_match][0]
                            pts[spatialStart:spatialEnd] = False
                else:
                    pts = ptsAll

                if np.sum(np.isfinite(dep_var[pts])) < pts_threshold:
                    dep_var_model = np.zeros_like(ind_var)
                else:
                    polyFit = phot_pipeline.robust_poly(ind_var[pts],dep_var[pts],fitOrder,
                                                        preScreen=self.param['backPreScreen'])
                    dep_var_model = np.polyval(polyFit,ind_var)
                
                if oneDirection == 'X':
                    bkgModel[cross_Ind,ind_var[ptsToSubtract]] = dep_var_model[ptsToSubtract]
                else:
                    bkgModel[ind_var[ptsToSubtract],cross_Ind] = dep_var_model[ptsToSubtract]
                
                if showEach == True:
                    plt.plot(ind_var,dep_var,label='data')
                    plt.plot(ind_var[pts],dep_var[pts],'o',color='red',label='pts fit')
                    plt.plot(ind_var,dep_var_model,label='model')
                    plt.show()
                    pdb.set_trace()
        
        outHead = deepcopy(head)
        if oneDirection == 'X':
            outHead['ROWSUB'] = (True, "Is row-by-row subtraction performed?")
        else:
            outHead['COLSUB'] = (True, "Is col-by-col subtraction performed?")
        
        if saveFits == True:
            primHDU = fits.PrimaryHDU(img,head)
            if custPrefix is not None:
                prefixName = custPrefix
            elif ind == None:
                prefixName = 'unnamed'
            else:
                prefixName = os.path.splitext(os.path.basename(self.fileL[ind]))[0]
            origName = '{}_for_backsub_{}.fits'.format(prefixName,oneDirection)
            origPath = os.path.join(self.baseDir,'diagnostics','spec_backsub',origName)
            primHDU.writeto(origPath,overwrite=True)
            primHDU_mod = fits.PrimaryHDU(bkgModel)
            
            subModelName = '{}_backsub_model_{}.fits'.format(prefixName,oneDirection)
            subModelPath = os.path.join(self.baseDir,'diagnostics','spec_backsub',subModelName)
            primHDU_mod.writeto(subModelPath,overwrite=True)
            
            subName = '{}_subtracted_{}.fits'.format(prefixName,oneDirection)
            subPath = os.path.join(self.baseDir,'diagnostics','spec_backsub',subName)
            subHDU = fits.PrimaryHDU(img - bkgModel,outHead)
            subHDU.writeto(subPath,overwrite=True)
            
        
        return img - bkgModel, bkgModel, outHead
    
    def do_backsub(self,img,head,ind=None,saveFits=False,directions=['Y','X'],
                   custPrefix=None):
        """
        Do all background subtractions
        
        Parameters
        ----------
        img: numpy array
            The image do to background subtraction on
        head: astropy.fits.header object
            The header of the image
        ind: int, or NOne
            The index of the file list.
            This is used to name diagnostic images, if requested
        saveFits: bool
            Save diagnostic FITS images of the background subtracted steps?
        directions: list of str
            The directions to extract, such as ['Y','X'] to subtract along Y
            and then X
        custPrefix: str or None
            A prefix for the output file name if saving diagnostic files
        
        Returns
        -------
        subImg: numpy array
            Background subtracted image
        bkgModelTotal: numpy array
            The background model
        subHead: astropy.fits.header object
            A header for the background-subtracted image
        """
        subImg = img
        subHead = head
        bkgModelTotal = np.zeros_like(subImg)
        for oneDirection in directions:
            subImg, bkgModel, subHead = self.backsub_oneDir(subImg,subHead,oneDirection,
                                                            ind=ind,saveFits=saveFits,
                                                            custPrefix=custPrefix)
            bkgModelTotal = bkgModelTotal + bkgModel
        return subImg, bkgModelTotal, subHead
    
    def save_one_backsub(self,ind):
        """
        Save a background-subtracted 
        """
        if np.mod(ind,15) == 0:
            print("On {} of {}".format(ind,len(self.fileL)))
        
        oneImgName = self.fileL[ind]
        img, head = self.getImg(oneImgName)
        
        imgSub, bkgModel, subHead = self.do_backsub(img,head,saveFits=False,
                                                    directions=self.param['bkgSubDirections'])
        outHDU = fits.PrimaryHDU(imgSub,subHead)
        baseName = os.path.splitext(os.path.basename(oneImgName))[0]
        outName = os.path.join(self.backsubDir,baseName+'.fits')
        outHDU.writeto(outName)
        
    def check_trace_requirements(self):
        """
        Some requirements for the trace code to work
        """
        if self.param['traceCurvedSpectrum'] == True:
            numSubtractions = len(self.param['bkgSubDirections'])
            assert numSubtractions <= 1,'<=1 background subtraction allowed currently'
            if numSubtractions == 1:
                if self.param['dispDirection'] == 'x':
                    assert (self.param['bkgSubDirections'][0] == 'Y'),'x dispersion must have Y backsub'
                    assert (len(self.param['bkgRegionsY']) == 1),'Must have 1 subtraction only'
                else:
                    assert (self.param['bkgSubDirections'][0] == 'X'),'y dispersion must have X backsub'
                    assert (len(self.param['bkgRegionsX']) == 1),'Must have 1 subtraction only'
        if self.traceReference is None:
            pass
        else:
            assert os.path.exists(self.traceReference), 'Trace Reference not found'

    def eval_trace(self,dispArray,src=0):
        """
        Evaluate the trace function
        
        Parameters
        ------------
        dispArray: numpy array
            Dispersion array in pixels
        """
        if self.param['traceCurvedSpectrum'] == True:
            self.find_trace()
            poly1 = self.traceInfo['poly {}'.format(src)]
            spatial_position = np.polyval(poly1,np.array(dispArray))
        else:
            spatial_position = self.param['starPositions'][src] * np.ones_like(dispArray,dtype=int)
        return spatial_position
        

    def find_trace(self,recalculateTrace=False,
                   recalculateTraceData=False,
                   showPlot=False):
        """
        Find the trace either from saved file or trace data

        Parameters
        -----------
        recalculateTrace: bool
            Recalculate the trace?
        recalculateTraceData: bool
            Recalculate the trace data?
        showPlot: bool
            Show plot of fitting the trace?
        """
        if (os.path.exists(self.traceFile) == True) & (recalculateTrace == False):
            pass
        else:
            if (os.path.exists(self.traceDataFile) == True) & (recalculateTraceData == False):
                pass
            else:
                img, head = self.get_default_im()
                res = self.fit_trace(img,head)
                res.write(self.traceDataFile,overwrite=True)
            traceData = ascii.read(self.traceDataFile)
            
            tpoly = Table()
            traceOrder = self.param['traceOrder']
            tpoly['order'] = np.flip(np.arange(traceOrder+1))
            for oneSrc in np.arange(self.nsrc):
                tracePoly = phot_pipeline.robust_poly(traceData['x'],
                                                      traceData['cen {}'.format(oneSrc)],
                                                      traceOrder)
                tpoly['poly {}'.format(oneSrc)] = tracePoly
            tpoly.write(self.traceFile,overwrite=True)

        self.traceInfo = ascii.read(self.traceFile)
        if showPlot == True:
            traceData = ascii.read(self.traceDataFile)
            
            for oneSrc in np.arange(self.nsrc):
                plt.plot(traceData['x'],traceData['cen {}'.format(oneSrc)],'o')
                y_eval = self.eval_trace(traceData['x'],src=oneSrc)
                plt.plot(traceData['x'],y_eval)
            plt.show()

    def fit_trace(self,img,head,showEach=False,
                   fitMethod='astropy'):
        """
        Use Gaussian to fit trace and FWHM

        Parameters
        ----------
        img: numpy array
            2D image to fit trace for
        head: astropy header
            FITS header to do fitting on
        showEach: bool
            Show each line fit?
        fitMethod: str
            'astropy' : will use models Gaussian2D + linear
            'sckpyQuick' : will fit w/ scipy.stats.norm
        """
        dispDirection = self.param['dispDirection'].upper()
        if dispDirection == 'Y':
            spatialIndexArrayLength = img.shape[1]
#            dispersionIndexArrayLength = img.shape[0]
        elif dispDirection == 'X':
            spatialIndexArrayLength = img.shape[0]
 #           dispersionIndexArrayLength = img.shape[1]
        else:
            raise Exception("Unrecognized spatial direction")
        ## eventually use self.param['dispOffsets'][srcInd]
        dispStart, dispEnd = self.param['dispPixels']
        dispersionIndexArrayLength = int(dispEnd - dispStart)

        n_spatials = self.nsrc
        # if self.param['mosBacksub'] == True:
        #     n_spatials = self.nsrc ## separate backsub for each source
        # else:
        #     n_spatials = 1 ## one backsub for the whole image
        
        fitOrder = self.param['traceOrder']
        ## make a source model
        srcModel = np.zeros_like(img)
        
        ## save the FWHM and centroid
        fwhmArr = np.zeros([dispersionIndexArrayLength,self.nsrc])
        cenArr = np.zeros([dispersionIndexArrayLength,self.nsrc])
        
        ## loop through the number of spatials
        for srcInd in np.arange(n_spatials):
            
            if self.param['mosBacksub'] == True:
                raise NotImplementedError
                
            else:
                dispersionIndexArray = np.arange(dispStart,dispEnd)
                spatialIndexArray = np.arange(spatialIndexArrayLength)
                ## fit the whole row/column
                ptsTofit = np.ones(len(spatialIndexArray),dtype=bool)

                dispersionCounterArray = np.arange(dispersionIndexArrayLength)
            
            ## set up which points to do source fitting for
            pts = np.zeros(len(spatialIndexArray),dtype=bool)
            srcMid = self.param['starPositions'][srcInd]
            boxSize = self.param['traceFitBoxSize']
            startsrc = np.max([0,int(srcMid - boxSize)])
            endsrc = np.min([np.max(spatialIndexArray),int(srcMid + boxSize)])
            
            pts[startsrc:endsrc] = True

            
            for dispersion_counter in tqdm.tqdm(dispersionCounterArray):
                dispersion_Ind = dispersionIndexArray[dispersion_counter]
                ind_var = spatialIndexArray ## independent variable
                if dispDirection == 'Y':
                    dep_var = img[dispersion_Ind,:]
                else:
                    dep_var = img[:,dispersion_Ind]

                
                good_pts = np.isfinite(dep_var) & pts
                spatial_profile = dep_var[good_pts]
                if np.sum(good_pts) < 5:
                    cenArr[dispersion_counter,srcInd] = np.nan
                    fwhmArr[dispersion_counter,srcInd] = np.nan
                    dep_var_model = np.ones_like(dep_var) * np.nan
                elif fitMethod == 'astropy':
                    fitter = fitting.LevMarLSQFitter()
                    ampGuess = np.percentile(spatial_profile,90)
                    meanGuess = spatialIndexArray[good_pts][np.argmax(spatial_profile)]
                    #meanGuess = np.sum(spatial_profile * spatialIndexArray[good_pts])/np.sum(spatial_profile)
                    gauss1d = models.Gaussian1D(amplitude=ampGuess, mean=meanGuess, 
                                                stddev=self.param['traceFWHMguess']/2.35)
                    line_orig = models.Linear1D(slope=0.0, intercept=np.min(spatial_profile))
                    comb_gauss = line_orig + gauss1d
                    try:
                        fitted_model = fitter(comb_gauss, ind_var[good_pts],spatial_profile, maxiter=111)
                    except Exception as e:
                        pdb.set_trace()
                    
                    cen_found = fitted_model.mean_1.value
                    cenGrace = 5
                    if (cen_found > (np.min(ind_var) - cenGrace)) & (cen_found < (np.max(ind_var) + cenGrace)):
                        thisCen = cen_found
                    else:
                        thisCen = np.nan

                    cenArr[dispersion_counter,srcInd] = thisCen
                    fwhmArr[dispersion_counter,srcInd] = fitted_model.stddev_1.value * 2.35
                    dep_var_model = fitted_model
                elif fitMethod == 'scipyQuick':
                    mean,std = norm.fit(ind_var,spatial_profile)
                    cenArr[dispersion_counter,srcInd] = mean
                    fwhmArr[dispersion_counter,srcInd] = std * 2.35
                else:
                    raise Exception("No fit method {}".format(fitMethod))
                # polyFit = phot_pipeline.robust_poly(ind_var[pts],dep_var[pts],fitOrder,
                #                                     preScreen=self.param['backPreScreen'])
                # dep_var_model = np.polyval(polyFit,ind_var)
                
                # if dispDirection == 'Y':
                #     bkgModel[dispersion_Ind,ind_var[ptsTofit]] = dep_var_model[ptsTofit]
                # else:
                #     bkgModel[ind_var[ptsTofit],dispersion_Ind] = dep_var_model[ptsTofit]
                
                if showEach == True:
                    plt.plot(ind_var,dep_var,label='data')
                    plt.plot(ind_var[good_pts],dep_var[good_pts],'o',color='red',label='pts fit')
                    plt.plot(ind_var,dep_var_model(ind_var),label='model')
                    plt.show()
                    pdb.set_trace()
                disperion_counter = dispersion_counter + 1

        t = Table()
        t['x'] = dispersionIndexArray
        for oneSrc in np.arange(self.nsrc):
            t['cen {}'.format(oneSrc)] = cenArr[:,oneSrc]
            t['fwhm {}'.format(oneSrc)] = fwhmArr[:,oneSrc]
        return t

    def save_all_backsub(self,useMultiprocessing=False):
        """
        Save all background-subtracted images
        """
        fileCountArray = np.arange(self.nImg)
        
        oneImgPath = self.fileL[0]
        self.backsubDir = os.path.join(os.path.split(oneImgPath)[0],'backsub_img')
        if os.path.exists(self.backsubDir) == False:
            os.mkdir(self.backsubDir)
        
        if useMultiprocessing == True:
            outputSpec = phot_pipeline.run_multiprocessing_phot(self,fileCountArray,method='save_one_backsub')
        else:
            outputSpec = []
            for ind in fileCountArray:
                outputSpec.append(self.save_one_backsub(ind))
        
    
    def profile_normalize(self,img,method='sum'):
        """
        Renormalize a profile along the spatial direction
        
        Parameters
        -----------
        img: numpy array
            The input profile image to be normalized
        
        """
        if method == 'sum':
            normArr = np.nansum(img,self.spatialAx)
        elif method == 'peak':
            normArr = np.nanmax(img,self.spatialAx)
        else:
            raise Exception("Unrecognized normalization method")
        
        if self.param['dispDirection'] == 'x':
            norm2D = np.tile(normArr,[img.shape[0],1])
        else:
            norm2D = np.tile(normArr,[img.shape[1],1]).transpose()
        
        norm_profile = np.zeros_like(img)
        nonZero = img != 0
        norm_profile[nonZero] = img[nonZero]/norm2D[nonZero]
        return norm_profile
    
    def find_profile(self,img,head,ind=None,saveFits=False,showEach=False,masterProfile=False):
        """
        Find the spectroscopic profile using splines along the spectrum
        This assumes an inherently smooth continuum (like a stellar source)
        
        img: numpy array
            The 2D Science image
        head: astropy.io.fits header object
            Header from the science file
        ind: int
            Index of the file list (which image is begin analyzed)
        saveFits: bool (optional)
            Save the profile to a fits file?
        showEach: bool
            Show each step of the profile fitting
        masterProfile: bool
            Is this a master profile fit?
        """
        
        profile_img_list = []
        smooth_img_list = [] ## save the smooth version if running diagnostics
        
        for srcInd,oneSourcePos in enumerate(self.param['starPositions']):
            dispStart = int(self.param['dispPixels'][0] + self.dispOffsets[srcInd])
            dispEnd = int(self.param['dispPixels'][1] + self.dispOffsets[srcInd])
            
            ind_var = np.arange(dispStart,dispEnd) ## independent variable
            knots = np.linspace(dispStart,dispEnd,self.param['numSplineKnots'])[1:-1]
            
            
            profile_img = np.zeros_like(img)
            if (self.param['traceCurvedSpectrum'] == True):
                t = self.calculate_apertures(src=srcInd)
                startSpatial = np.min(t['srcStart'])
                endSpatial = np.max(t['srcEnd'])
            else:
                startSpatial = int(oneSourcePos - self.param['apWidth'] / 2.)
                endSpatial = int(oneSourcePos + self.param['apWidth'] / 2.)
            
            for oneSpatialInd in np.arange(startSpatial,endSpatial + 1):
                if self.param['dispDirection'] == 'x':
                    try:
                        dep_var = img[oneSpatialInd,dispStart:dispEnd]
                    except IndexError:
                        print("indexing problem. Entering pdb to help diagnose it")
                        pdb.set_trace()
                else:
                    dep_var = img[dispStart:dispEnd,oneSpatialInd]
                
                ## this is a very long way to calculate a log that avoids runtime warnings
                fitY = np.zeros_like(dep_var) * np.nan
                positivep = np.zeros_like(dep_var,dtype=bool)
                finitep = np.isfinite(dep_var)
                positivep[finitep] = (dep_var[finitep] > 0. - self.floor_delta)
                fitY[positivep] = np.log10(dep_var[positivep] + self.floor_delta)
                
                spline1 = phot_pipeline.robust_poly(ind_var,fitY,self.param['splineSpecFitOrder'],
                                                    knots=knots,useSpline=True,sigreject=self.param['splineSigRej'],
                                                    plotEachStep=False,preScreen=self.param['splinePreScreen'])
                
                modelF = 10**spline1(ind_var) - self.floor_delta
                
                if showEach == True:
                    plt.plot(ind_var,dep_var,'o',label='data')
                    plt.plot(ind_var,modelF,label='model')
                    yKnots = np.nanmedian(dep_var) * np.ones_like(knots)
                    plt.plot(knots,yKnots,'o',label='knots')
                    plt.legend()
                    plt.show()
                    pdb.set_trace()
                
                if self.param['dispDirection'] == 'x':
                    profile_img[oneSpatialInd,dispStart:dispEnd] = modelF
                else:
                    profile_img[dispStart:dispEnd,oneSpatialInd] = modelF
            
            smooth_img = deepcopy(profile_img)

            if (self.param['traceCurvedSpectrum'] == True):
                ## set the pixels below and above the curved trace to be 0 in profile
                oneTable = self.calculate_apertures(src=srcInd)
                for oneDispPx in np.arange(dispStart,dispEnd):
                    disp_pts_match = (oneDispPx == oneTable['dispersion_px'])
                    if np.sum(disp_pts_match) > 0:
                        local_spatialStart = oneTable['srcStart'][disp_pts_match][0]
                        local_spatialEnd = oneTable['srcEnd'][disp_pts_match][0]
                        
                        if self.param['dispDirection'] == 'x':
                            profile_img[startSpatial:local_spatialStart,oneDispPx] = 0
                            profile_img[local_spatialEnd:(endSpatial+1),oneDispPx] = 0
                            profile_img[local_spatialStart:(local_spatialEnd+1),oneDispPx] += self.floor_delta
                        else:
                            profile_img[oneDispPx,startSpatial:local_spatialStart] = 0
                            profile_img[oneDispPx,(local_spatialEnd+1):endSpatial] = 0
                            profile_img[oneDispPx,local_spatialStart:(local_spatialEnd+1)] += self.floor_delta
            else:
                if self.param['dispDirection'] == 'x':
                    profile_img[startSpatial:endSpatial+1,dispStart:dispEnd] += self.floor_delta
                else:
                    profile_img[dispStart:dispEnd,startSpatial:endSpatial+1] += self.floor_delta
            
            ## Renormalize            
            norm_profile = self.profile_normalize(profile_img)
            profile_img_list.append(norm_profile)
            
            ## save the smoothed image
            smooth_img_list.append(smooth_img)

        if saveFits == True:
            primHDU = fits.PrimaryHDU(img,head)
            if masterProfile == True:
                prefixName = self.master_profile_prefix
            elif ind == None:
                prefixName = 'unnamed'
            else:
                prefixName = os.path.splitext(os.path.basename(self.fileL[ind]))[0]
            origName = '{}_for_profile_fit.fits'.format(prefixName)
            origPath = os.path.join(self.baseDir,'diagnostics','profile_fit',origName)
            primHDU.writeto(origPath,overwrite=True)
            for ind,profile_img in enumerate(profile_img_list):
                ## Saved the smoothed model
                primHDU_smooth = fits.PrimaryHDU(smooth_img_list[ind])
                smoothModelName = '{}/{}_smoothed_src_{}.fits'.format(self.profile_dir,prefixName,ind)
                primHDU_smooth.writeto(smoothModelName,overwrite=True)
                
                ## Save the profile
                primHDU_mod = fits.PrimaryHDU(profile_img)
                profModelName = '{}/{}_profile_model_src_{}.fits'.format(self.profile_dir,prefixName,ind)
                primHDU_mod.writeto(profModelName,overwrite=True)
        
        
        return profile_img_list, smooth_img_list
    

    def read_profiles(self):
        """ Read in the master profile for each source if using a single profile for all images """
        profile_img_list, smooth_img_list = [], []
        prefixName = self.master_profile_prefix
        for ind in np.arange(self.nsrc):
            ## Get the profile
            profModelName = '{}/{}_profile_model_src_{}.fits'.format(self.profile_dir,prefixName,ind)
            profile_img_list.append(fits.getdata(profModelName))
            
            ## Get the smoothed model
            smoothModelName = '{}/{}_smoothed_src_{}.fits'.format(self.profile_dir,prefixName,ind)
            smooth_img_list.append(fits.getdata(smoothModelName))
        
        return profile_img_list, smooth_img_list
    
    def find_cov_weights(self,nDisp,varImg,profile_img,readNoise,
                         src=0,saveWeights=False,diagnoseCovariance=False):
        
        ## This will be slow at first because I'm starting with a for loop.
        ## Eventually do fancy 3D matrices to make it fast
        #optflux = np.zeros(nDisp) * np.nan
        #varFlux = np.zeros_like(optflux) * np.nan
        varPure = varImg - readNoise**2 ## remove the read noise because we'll put it in covariance matrix
        weight2D = np.zeros_like(profile_img)
        
        if self.param['traceCurvedSpectrum'] == True:
            apTable = self.calculate_apertures(src=src)
        else:
            oneSourcePos = self.param['starPositions'][src]
            startSpatial = int(oneSourcePos - self.param['apWidth'] / 2.)
            endSpatial = int(oneSourcePos + self.param['apWidth'] / 2.)
            nSpatial = (endSpatial - startSpatial) + 1
        
        for oneInd in np.arange(nDisp):

            if self.param['traceCurvedSpectrum'] == True:
                disp_pts_match = (oneInd == apTable['dispersion_px'])
                if np.sum(disp_pts_match) > 0:
                    startSpatial = apTable['srcStart'][disp_pts_match][0]
                    endSpatial = apTable['srcEnd'][disp_pts_match][0]
                    nSpatial = (endSpatial - startSpatial) + 1
                else:
                    startSpatial = 1
                    endSpatial = 0 ##make it have zero points
                    nSpatial = 0
            
            if self.param['dispDirection'] == 'x':
                prof = profile_img[startSpatial:endSpatial+1,oneInd]
                varPhotons = varPure[startSpatial:endSpatial+1,oneInd]
                #correction = correctionFactor[oneInd]
                #data = imgSub[startSpatial:endSpatial+1,oneInd]
            else:
                prof = profile_img[oneInd,startSpatial:endSpatial+1]
                varPhotons = varPure[oneInd,startSpatial:endSpatial+1]
                #correction = correctionFactor[oneInd]
                #data = imgSub[oneInd,startSpatial:endSpatial]
            
            minP = self.minPixForCovarianceWeights
            if (np.nansum(prof > 0) > minP) & (np.sum(np.isfinite(varPhotons)) > minP):
                ## only try to do the matrix math if points are finite
                
                ## assuming the read noise is correlated but photon noise is not
                rho = self.param['readNoiseCorrVal']
                ## fill everything w/ off-diagonal
                cov_read = np.ones([nSpatial,nSpatial]) * rho * readNoise**2
                
                if self.param['dispNoiseCorrelation'] == True:
                    dispPixels = 100
                    ## fill diagonal w/ spatial correlation
                    np.fill_diagonal(cov_read,self.param['readNoiseCorrDispVal'] * readNoise**2)
                    cov_read = np.repeat(cov_read,dispPixels,axis=0)
                    cov_read = np.repeat(cov_read,dispPixels,axis=1)
                    
                    ## fill diagonal of spatial-spectral w/ read noise
                    np.fill_diagonal(cov_read,readNoise**2)
                    
                    prof = np.repeat(prof,dispPixels)
                    ## renormalize
                    prof = prof / np.sum(prof)
                    
                    varPhotons = np.repeat(varPhotons,dispPixels)
                else:
                    ## fill diagonal w/ read noise
                    np.fill_diagonal(cov_read,readNoise**2)
                
                if self.param['ignorePhotNoiseInCovariance'] == True:
                    ## A diagnostic mode to experiment with simulated data w/ no photon noise
                    cov_matrix = cov_read ## temporarily ignoring phot noise as in simulation
                else:
                    cov_matrix = np.diag(varPhotons) + cov_read
                
                cov_matrix_norm = np.outer(1./prof,1./prof) * cov_matrix
                
                inv_cov = np.linalg.inv(cov_matrix_norm)
                
                weights = np.dot(np.ones_like(prof),inv_cov)
                #optflux[oneInd] = np.nansum(weights * data * correction / prof) / np.sum(weights)
                #varFlux[oneInd] = np.nansum(correction) / np.sum(weights)
                if self.param['dispNoiseCorrelation'] == True:
                    x = np.arange(len(weights))
                    sumSpatial, edges, binnum = binned_statistic(x,weights,statistic='sum',bins=nSpatial)
                    weights = sumSpatial * nSpatial
                    
                    ## diagnostic stuff
                    # prof = profile_img[startSpatial:endSpatial+1,oneInd]
#                     var = varImg[startSpatial:endSpatial+1,oneInd]
#                     weights_var = (prof**2 / var)
#                     print('Weights var: ')
#                     print(weights_var)
#                     print("Weights cov:")
#                     print(weights)
#                     photons = np.sum(varPhotons)
#                     err_var = np.sqrt(1./np.sum(prof**2/var))
#                     print("Err var (e-, %):")
#                     print(err_var,err_var/photons * 100.)
#                     print("Err cov:")
#                     err_cov = np.sqrt(1./np.sum(weights))
#                     print(err_cov, err_cov/photons * 100.)
#                     pdb.set_trace()
                
                if (diagnoseCovariance == True) & (oneInd > 900):
                    var = varImg[startSpatial:endSpatial+1,oneInd]
                    weights_var = (prof / var) / np.sum(prof**2/var)
                    if self.param['dispNoiseCorrelation'] == True:
                        prof = profile_img[startSpatial:endSpatial+1,oneInd]
                    weights_cov = (weights / prof) / np.sum(weights)
                    print('Weights var:')
                    print(weights_var)
                    print('Weights covar:')
                    print(weights_cov)
                    pdb.set_trace()
                
                if self.param['dispDirection'] == 'x':
                    weight2D[startSpatial:endSpatial+1,oneInd] = weights
                else:
                    weight2D[oneInd,startSpatial:endSpatial+1] = weights
                
        if saveWeights == True:
            primHDU = fits.PrimaryHDU(weight2D)
            outName = '{}_weight2D_src_{}.fits'.format(self.dataFileDescrip,src)
            outPath = os.path.join(self.weight_dir,outName)
            primHDU.writeto(outPath,overwrite=True)
            
        
        return weight2D
    
    def read_cov_weights(self,src=0):
        """
        Read the covariance-weights for a fixed profile through the time series 
        """
        weightName = '{}_weight2D_src_{}.fits'.format(self.dataFileDescrip,src)
        weightPath = os.path.join(self.weight_dir,weightName)
        weight2D = fits.getdata(weightPath)
        return weight2D
    
    def spec_for_one_file(self,ind,saveFits=False,diagnoseCovariance=False):
        """ Get spectroscopy for one file
        Calculate the optimal and sum extractions
        
        If `saveRefRow` is True, the reference pixel row will be saved
        
        Parameters
        ----------
        ind: int
            File index
        saveFits: bool
            Save the background subtraction and profile fits?
        diagnoseCovariance: bool
            Diagnostic information for the covariance profile weighting
        """
        
        oneImgName = self.fileL[ind]
        img, head = self.getImg(oneImgName)
        t0 = self.get_date(head)
        
        if 'AIRMASS' in head:
            airmass = head['AIRMASS']
        else:
            airmass = 'none'
        
        imgSub, bkgModel, subHead = self.do_backsub(img,head,ind,saveFits=saveFits,
                                                    directions=self.param['bkgSubDirections'])
        readNoise = self.get_read_noise(head)
        ## Background and read noise only.
        ## Smoothed source flux added below
        varImg = readNoise**2 + np.abs(bkgModel) ## in electrons because it should be gain-corrected
        
        if self.param['fixedProfile'] == True:
            profile_img_list, smooth_img_list = self.read_profiles()
        else:
            profile_img_list, smooth_img_list = self.find_profile(imgSub,subHead,ind,saveFits=saveFits)
        
        for oneSrc in np.arange(self.nsrc): ## use the smoothed flux for the variance estimate
            varImg = varImg + np.abs(smooth_img_list[oneSrc]) ## negative flux is approximated as photon noise
        
        if saveFits == True:
            prefixName = os.path.splitext(os.path.basename(oneImgName))[0]
            varName = '{}_variance.fits'.format(prefixName)
            varPath = os.path.join(self.baseDir,'diagnostics','variance_img',varName)
            primHDU = fits.PrimaryHDU(varImg)
            primHDU.writeto(varPath,overwrite=True)
        
        spatialAx = self.spatialAx
        dispAx = self.dispAx
                
        ## dispersion indices in pixels (before wavelength calibration)
        nDisp = img.shape[dispAx]
        dispIndices = np.arange(nDisp)
        
        optSpectra = np.zeros([self.nsrc,nDisp])
        optSpectra_err = np.zeros_like(optSpectra)
        sumSpectra = np.zeros_like(optSpectra)
        sumSpectra_err = np.zeros_like(optSpectra)
        backSpectra = np.zeros_like(optSpectra)
        peakSpectra = np.zeros_like(optSpectra)
        rssSpectra = np.zeros_like(optSpectra) ## root sum square

        cenInfo = np.zeros([self.nsrc])
        fwhmInfo = np.zeros([self.nsrc])
        
        for oneSrc in np.arange(self.nsrc):
            profile_img = profile_img_list[oneSrc]
            smooth_img = smooth_img_list[oneSrc]
            
            ## Find the bad pixels and their missing weights
            finitep = (np.isfinite(img) & np.isfinite(varImg) & np.isfinite(smooth_img))
            badPx = finitep == False ## start by marking NaNs as bad pixels
            ## also mark large deviations from profile fit
            badPx[finitep] = np.abs(smooth_img[finitep] - imgSub[finitep]) > self.param['sigForBadPx'] * np.sqrt(varImg[finitep])
            holey_profile = deepcopy(profile_img)
            holey_profile[badPx] = 0.
            holey_weights = np.sum(holey_profile,self.spatialAx)
            correctionFactor = np.ones_like(holey_weights)
            goodPts = holey_weights > 0.
            correctionFactor[goodPts] = 1./holey_weights[goodPts]
            
            ## make a 2D correction factor to weight the image around NaNs
            if self.param['dispDirection'] == 'x':
                correct2D = np.tile(correctionFactor,[img.shape[0],1])
            else:
                correct2D = np.tile(correctionFactor,[img.shape[1],1]).transpose()
            markBad = badPx & (profile_img > 0.)
            correct2D[markBad] = 0.
            
            if saveFits == True:
                primHDU_prof2Dh = fits.PrimaryHDU(holey_profile)
                holey_profile_name = '{}_holey_profile_{}.fits'.format(prefixName,oneSrc)
                holey_profile_path = os.path.join(self.baseDir,'diagnostics','profile_fit',holey_profile_name)
                primHDU_prof2Dh.writeto(holey_profile_path,overwrite=True)
                
                corrHDU = fits.PrimaryHDU(correct2D)
                correction2DName = '{}_correct_2D_{}.fits'.format(prefixName,oneSrc)
                correction2DPath = os.path.join(self.baseDir,'diagnostics','profile_fit',correction2DName)
                corrHDU.writeto(correction2DPath,overwrite=True)
            
            srcMask = profile_img > 0.
            
            ## Replaced the old lines to avoid runtime warnings
            
            if self.param['readNoiseCorrelation'] == True:
                #if self.param['fixedProfile'] == True:
                fixedProfImplemented  = True
                if (self.param['fixedProfile'] == True) & (fixedProfImplemented == True):
                    weight2D = self.read_cov_weights(src=oneSrc)
                else:
                    weight2D = self.find_cov_weights(nDisp,varImg,profile_img,readNoise,
                                                     src=oneSrc,saveWeights=False,
                                                     diagnoseCovariance=diagnoseCovariance)
                
                inverse_prof = np.zeros_like(profile_img) * np.nan
                nonz = (profile_img != 0.) & (np.isfinite(profile_img))
                inverse_prof[nonz] = 1./profile_img[nonz]
                # avoid div by 0 issues
                optNumerator = np.nansum(imgSub * weight2D * correct2D * inverse_prof,spatialAx)
                denom = np.nansum(weight2D,spatialAx)
                varNumerator = correctionFactor
                denom_v = denom
            
            elif self.param['superWeights'] == True:
                expConst = 10.
                weight2D = profile_img * np.exp(profile_img * expConst)/ varImg
                #weight2D = self.profile_normalize(weight2D,method='peak')
                #normprof2 = self.profile_normalize(profile_img,method='peak') * np.median(np.nanmax(profile_img,spatialAx))
                optNumerator = np.nansum(imgSub * weight2D,spatialAx)
                denom =  np.nansum(profile_img * weight2D,spatialAx)
                varNumerator = np.nansum(profile_img * correct2D,spatialAx)
                denom_v = np.nansum(profile_img**2/varImg,spatialAx)
            else:
                # optflux = (np.nansum(imgSub * profile_img * correct2D/ varImg,spatialAx) /
                #            np.nansum(profile_img**2/varImg,spatialAx))
                # varFlux = (np.nansum(profile_img * correct2D,spatialAx) /
                #            np.nansum(profile_img**2/varImg,spatialAx))
                optNumerator = np.nansum(imgSub * profile_img * correct2D/ varImg,spatialAx)
                denom =  np.nansum(profile_img**2/varImg,spatialAx)
                varNumerator = np.nansum(profile_img * correct2D,spatialAx)
                denom_v = denom
            
            nonz = (denom != 0.) & np.isfinite(denom)
            optflux = np.zeros_like(optNumerator) * np.nan
            optflux[nonz] = optNumerator[nonz] / denom[nonz]
            
            varFlux = np.zeros_like(varNumerator) * np.nan
            varFlux[nonz] = varNumerator[nonz] / denom_v[nonz]
            
            sumFlux = np.nansum(imgSub * srcMask,spatialAx)
            sumErr = np.sqrt(np.nansum(varImg * srcMask,spatialAx))
            
            optSpectra[oneSrc,:] = optflux
            optSpectra_err[oneSrc,:] = np.sqrt(varFlux)
            
            sumSpectra[oneSrc,:] = sumFlux
            sumSpectra_err[oneSrc,:] = sumErr
            
            backSpectra[oneSrc,:] = np.nanmean(bkgModel * srcMask,spatialAx)

            ## save the peak spectra, removing bad px adding back in background model
            holey_img = deepcopy((imgSub + bkgModel) * srcMask)
            holey_img[badPx] = 0
            
            peakSpectra[oneSrc,:] = np.nanmax(holey_img,axis=spatialAx)
            rssSpectra[oneSrc,:] = np.sqrt(np.nansum(holey_img**2,axis=spatialAx))

            if saveFits == True:
                prefixName = os.path.splitext(os.path.basename(oneImgName))[0]
                if self.param['readNoiseCorrelation'] == True:
                    pass ## already using the weight2D
                    weightName = '_rn_corr'
                elif self.param['superWeights'] == True:
                    weight2D = weight2D
                    weightName = '_super'
                else:
                    ## calculate weight 2D
                    weight2D = profile_img * correctionFactor/ varImg
                    weightName = ''
                
                weightName = '{}_weights{}.fits'.format(prefixName,weightName)
                weightPath = os.path.join(self.baseDir,'diagnostics','variance_img',weightName)
                primHDU = fits.PrimaryHDU(weight2D)
                primHDU.writeto(weightPath,overwrite=True)
            
            if self.param['saveSpatialProfileStats'] == True:
                ## get the centroid and FWHM info
                if self.param['profilePix'] == None:
                    profDispStart_0 = self.param['dispPixels'][0]
                    profDispEnd_0 = self.param['dispPixels'][1]
                else:
                    profDispStart_0 = self.param['profilePix'][0]
                    profDispEnd_0 = self.param['profilePix'][1]
                
                profDispStart = profDispStart_0 + self.dispOffsets[oneSrc]
                profDispEnd = profDispEnd_0 + self.dispOffsets[oneSrc]
                
                profilePix = ((dispIndices > profDispStart) &
                             (dispIndices <= profDispEnd))
                
                oneSourcePos = self.param['starPositions'][oneSrc]
                if self.param['profileFitWidth'] is None:
                    spatialWidthToFit = self.param['apWidth']
                else:
                    spatialWidthToFit = self.param['profileFitWidth']
                
                startSpatial = int(oneSourcePos - spatialWidthToFit / 2.)
                endSpatial = int(oneSourcePos + spatialWidthToFit / 2.)
                spatial_var = np.arange(startSpatial,endSpatial) ## independent variable
                
                if self.param['useSmoothProfileForStats'] == True:
                    if self.param['dispDirection'] == 'x':
                        spatial_profile = np.nanmean(profile_img[startSpatial:endSpatial,profilePix],axis=1)
                    else:
                        spatial_profile = np.nanmean(profile_img[profilePix,startSpatial:endSpatial],axis=0)
                else:
                    if self.param['dispDirection'] == 'x':
                        spatial_profile = np.nanmedian(imgSub[startSpatial:endSpatial,profilePix],axis=1)
                    else:
                        spatial_profile = np.nanmedian(imgSub[profilePix,startSpatial:endSpatial],axis=0)
                
                fitter = fitting.LevMarLSQFitter()
                #fitter = fitting.DogBoxLSQFitter()
                ampGuess = np.max(spatial_profile) - np.min(spatial_profile)
                meanGuess = np.median(spatial_var)

                gauss1d = models.Gaussian1D(amplitude=ampGuess, mean=meanGuess,
                                            stddev=self.param['traceFWHMguess']/2.35,
                                            bounds={"amplitude":(0.,np.inf),
                                                    "stddev":(0.2,np.inf),
                                                    "mean":(np.min(spatial_var)-1.,
                                                            np.max(spatial_var)+1.)})
                line_orig = models.Linear1D(slope=0.0, intercept=np.min(spatial_profile))
                comb_gauss = line_orig + gauss1d
                fitted_model = fitter(comb_gauss, spatial_var,spatial_profile, maxiter=111)
                cenInfo[oneSrc] = fitted_model.mean_1.value
                fwhmInfo[oneSrc] = fitted_model.stddev_1.value * 2.35
                if saveFits == True:
                    fig, ax = plt.subplots()
                    ax.plot(spatial_var,spatial_profile,label='data')
                    ax.plot(spatial_var,comb_gauss(spatial_var),label='Initial Guess')
                    ax.plot(spatial_var,fitted_model(spatial_var),label='Model')
                    spat_profile_name = 'spatial_prof_{}_src_{}.pdf'.format(prefixName,oneSrc)
                    spatial_prof_path = os.path.join(self.baseDir,'diagnostics','spatial_profile',spat_profile_name)
                    ax.legend()
                    ax.set_title('Cen={}, FWHM={}'.format(cenInfo[oneSrc],fwhmInfo[oneSrc]))
                    print("Saving spatial profile to {}".format(spatial_prof_path))
                    fig.savefig(spatial_prof_path)
                    plt.close(fig)
        
        extractDict = {} ## spectral extraction dictionary
        extractDict['t0'] = t0
        extractDict['disp indices'] = dispIndices
        extractDict['opt spec'] = optSpectra
        extractDict['opt spec err'] = optSpectra_err
        extractDict['sum spec'] = sumSpectra
        extractDict['sum spec err'] = sumSpectra_err
        extractDict['back spec'] = backSpectra
        extractDict['airmass'] = airmass
        extractDict['peak spec'] = peakSpectra
        extractDict['rss spec'] = rssSpectra
        
        if self.param['saveSpatialProfileStats'] == True:
            extractDict['cen'] = cenInfo
            extractDict['fwhm'] = fwhmInfo
        
        if self.param['saveRefRow'] == True:
            refRow = np.mean(img[0:4,:],axis=0)
            extractDict['ref row'] = refRow
        
        return extractDict
    
    def norm_spec(self,x,y,numSplineKnots=None,srcInd=0):
        """ Normalize spec 
        
        Parameters
        ----------
        """
        dispStart = self.param['dispPixels'][0] + self.dispOffsets[srcInd]
        dispEnd = self.param['dispPixels'][1] + self.dispOffsets[srcInd]
        if numSplineKnots is None:
            numSplineKnots = self.param['numSplineKnots']
        
        knots = np.linspace(dispStart,dispEnd,numSplineKnots)[1:-1]
        spline1 = phot_pipeline.robust_poly(x,np.log10(y),self.param['splineSpecFitOrder'],
                                            knots=knots,useSpline=True)
        modelF = 10**spline1(x)
        return y / modelF
    
    def plot_one_spec(self,src=0,ind=None,specTypes=['Sum','Optimal'],
                      normalize=False,numSplineKnots=None,showPlot=True,waveCal=False):
        """
        Plot one example spectrum after extraction has been run
        
        Parameters
        ----------
        src: int, optional
            The number of the source to plot
        ind: int or None, optional
            An index number to pass to get_spec().
            It tells which spectrum to plot.
            Defaults to number of images //2
        specTypes: list of strings, optional
            List of which spectra to show
            'Sum' for sum extraction
            'Optimal' for optimal extraction
        normalize: bool, optional
            Normalize and/or flatten spectrum using `self.norm_spec`
        numSplitKnots: int or None, optional
            Number of spline knots to pass to `self.norm_spec` for flattening
        showPlot: bool
            Show the plot in widget?
            If True, it renders the image with plt.show()
            If False, saves an image
        waveCal: bool
            Wavelength calibrate the X axis? If False, gives dispersion pixels
        """
        fig, ax = plt.subplots()
        
        for oneSpecType in specTypes:
            x, y, yerr = self.get_spec(specType=oneSpecType,ind=ind,src=src)
            if normalize==True:
                y = self.norm_spec(x,y,numSplineKnots=numSplineKnots)
            if waveCal == True:
                plot_x = self.wavecal(x)
            else:
                plot_x = x
            
            ax.plot(plot_x,y,label=oneSpecType)
        ax.legend()
        if waveCal == True:
            ax.set_xlabel(r"Wavelength ($\mu$m)")
        else:
            ax.set_xlabel("{} pixel".format(self.param['dispDirection'].upper()))
        
        ax.set_ylabel("Counts (e$^-$)")
        if showPlot == True:
            plt.show()
        else:
            outName = '{}_ind_spec_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
            outPath = os.path.join(self.baseDir,'plots','spectra','individual_spec',outName)
            fig.savefig(outPath,bbox_inches='tight')
        
        plt.close(fig)
        
    def periodogram(self,src=0,ind=None,specType='Optimal',showPlot=True,
                    transform=None,trim=False,logY=True,align=False):
        """
        Plot a periodogram of the spectrum to search for periodicities
                    
        Parameters
        -----------
        showPlot: bool
            If True, a plot is rendered in matplotlib widget
            If False, a plot is saved to file
        """
        
        if ind == None:
            x_px, y, yerr = self.get_avg_spec(src=src,align=align)
        else:
            x_px, y, yerr = self.get_spec(specType=specType,ind=ind,src=src)
        
        if trim == True:
            x_px = x_px[1000:1800]
            y = y[1000:1800]
            yerr = yerr[1000:1800]
            # x_px = x_px[1000:1200]
            # y = y[1000:1200]
            # yerr = yerr[1000:1200]
        
        if transform is None:
            x = x_px
            x_unit = 'Frequency (1/px)'
        elif transform == 'lam-squared':
            lam = self.wavecal(x_px)
            x = lam**2
            x_unit = r'Frequency (1/$\lambda^2$)'
        elif transform == 'inv-lam':
            lam = self.wavecal(x_px)
            x = 1./lam
            x_unit = r'$\lambda^2/(\Delta \lambda) (\mu$m)'
        else:
            raise Exception("Unrecognized transform {}".format(transform))
            
        
        normY = self.norm_spec(x_px,y,numSplineKnots=200)
        yerr_Norm = yerr / y
        #x1, x2 = 
        pts = np.isfinite(normY) & np.isfinite(yerr_Norm)
        ls = LombScargle(x[pts],normY[pts],yerr_Norm[pts])
        #pdb.set_trace()
        frequency, power = ls.autopower()
        period = 1./frequency
        
        fig, ax = plt.subplots()
        
        if logY == True:
            ax.loglog(frequency,power)
        else:
            ax.semilogx(frequency,power)
        
        ax.set_xlabel(x_unit)
        ax.set_ylabel('Power')
        
        if astropy.__version__ > "3.0":
            maxPower = power.max()
            
            fap = ls.false_alarm_probability(power.max())
            print("False alarm probability at {} is {}".format(maxPower,fap))
            for onePower in [1e-3,1e-2,2e-2,2.5e-2]:
                fap = ls.false_alarm_probability(onePower)
                print("False alarm probability at {} is {}".format(onePower,fap))
        else:
            warnings.warn('Not calculating FAP for older version of astropy')
        
        # localPts = frequency < 0.05
        # argmax = np.argmax(power[localPts])
        # freqAtMax = frequency[localPts][argmax]
        # print('Freq at local max power = {}'.format(freqAtMax))
        # print('Corresponding period = {}'.format(1./freqAtMax))
        # if astropy.__version__ > "3.0":
        #     print("FAP at local max = {}".format(ls.false_alarm_probability(power[localPts][argmax])))
        # else:
        #     warnings.warn('Not calculating FAP for older versions of astropy')
        
        #ax.set_xlim(100,1e4)
        
        if showPlot == True:
            plt.show()
        else:
            periodoName = '{}_spec_periodo_{}_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'],transform)
            outPath = os.path.join(self.baseDir,'plots','spectra','periodograms',periodoName)
            fig.savefig(outPath)
            plt.close(fig)
    
    def get_spec(self,specType='Optimal',ind=None,src=0):
        """
        Get a spectrum from the saved FITS file
        
        Parameters
        ------------
        specType: str
            The type of spectrum 'Optimal" for optimal extraction.
            'Sum' for sum extraction
        ind: int or None
            The index from the file list. If None, it will use the default.
        src: int
            The source index
        
        Returns
        --------
        x: numpy array
            The spectral pixel numbers
        y: numpy array
            the extracted flux
        yerr: numpy array
            the error in the extracted flux
        """
        if os.path.exists(self.specFile) == False:
            self.do_extraction()
        
        x, y, yerr = get_spectrum(self.specFile,specType=specType,ind=ind,src=src)
        return x, y, yerr
    
    def align_spec(self,data2D,refInd=None,diagnostics=False,srcInd=0):
        align2D = np.zeros_like(data2D)
        nImg = data2D.shape[0]
        dispPix = np.array(np.array(self.param['dispPixels']) + self.dispOffsets[srcInd],dtype=int)
        Noffset = self.param['nOffsetCC']
        
        if refInd == None:
            refInd = nImg // 2
        
        refSpec = data2D[refInd,dispPix[0]:dispPix[1]]
        waveIndices = np.arange(dispPix[1] - dispPix[0])
        
        offsetIndArr = np.zeros(nImg)
        
        for imgInd in np.arange(nImg):
            thisSpec = data2D[imgInd,dispPix[0]:dispPix[1]]
            if (imgInd > 5) & (diagnostics == True):
                doDiagnostics = True
            else:
                doDiagnostics = False
            
            offsetX, offsetInd = utils.crosscor_offset(waveIndices,refSpec,thisSpec,Noffset=Noffset,
                                                          diagnostics=doDiagnostics,subPixel=True,
                                                          lowPassFreq=self.param['lowPassFreqCC'],
                                                          highPassFreq=self.param['hiPassFreqCC'])
            if doDiagnostics == True:
                pdb.set_trace()
            
            align2D[imgInd,:] = utils.roll_pad(data2D[imgInd,:],offsetInd * self.param['specShiftMultiplier'])
            offsetIndArr[imgInd] = offsetInd * self.param['specShiftMultiplier']
        
        return align2D, offsetIndArr
    
    def get_avg_spec(self,src=0,redoDynamic=True,align=False):
        """
        Get the average spectrum across all of the time series
        
        Parameters
        ----------
        src: int
            The index number for the source (0 for the first source's spectrum)
        redoDynamic: bool
            Re-run the :any:`plot_dynamic_spec` method to save the average spectrum
        align: bool
            Pass this to the :any:`plot_dynamic_spec` method
        
        Returns
        -------
        x: numpy array
            the Wavelength dispersion values (in pixels)
        y: numpy array
            The Flux values of the average spectrum
        yerr: numpy array
            The error in the average flux
        """
        
        dyn_specFile = self.dyn_specFile(src=src)
        if (os.path.exists(dyn_specFile) == False) | (redoDynamic == True):
            self.plot_dynamic_spec(src=src,saveFits=True,align=align)
        
        HDUList = fits.open(dyn_specFile)
        x = HDUList['DISP INDICES'].data
        y = HDUList['AVG SPEC'].data
        yerr = HDUList['AVG SPEC ERR'].data
        
        HDUList.close()
        
        return x, y, yerr
    
    def get_one_peak_spec(self,src=0,ind=None):
        """
        Get the peak spectrum for one image
        """
        if ind == None:
            useInd = self.get_default_index()
        else:
            useInd = ind
        specRes = self.spec_for_one_file(useInd)
        t = Table()
        t['Disp Indices'] = specRes['disp indices']
        img, head = self.getImg(self.fileL[useInd])
        for oneSrc in np.arange(self.nsrc):
            cr_conversion = self.countrate_to_electrons_mult(head)
            t['Peak Spec {}'.format(oneSrc)] = specRes['peak spec'][oneSrc] / cr_conversion
            t['Rss Spec {}'.format(oneSrc)] = specRes['rss spec'][oneSrc] / cr_conversion
            t['Sum {}'.format(oneSrc)] = specRes['sum spec'][oneSrc] / cr_conversion
        t.write(self.peakSpecFile,overwrite=True)


    def dyn_specFile(self,src=0):
        return "{}_src_{}.fits".format(self.dyn_specFile_prefix,src)
        
    def plot_dynamic_spec(self,src=0,saveFits=True,specAtTop=True,align=False,
                          alignDiagnostics=False,extraFF=False,
                          specType='Optimal',showPlot=False,
                          vmin=None,vmax=None,flipX=False,
                          waveCal=False,topYlabel='',
                          interpolation=None,
                          smooth2D=None):
        """
        Plots a dynamic spectrum of the data
        
        Parameters
        -----------
        showPlot: bool, optional
            Show the plot in addition to saving?
        saveFits: bool
            Save a FITS file of the dynamic spectrum?
        specAtTop: bool
            Plot the average spectrum at the top of the dynamic spectrum?
        align: bool
            Automatically align all the spectra?
        alignDiagnostics: bool
            Show diagnostics of the alignment process?
        extraFF: bool
            Apply an extra flattening step?
        specType: str
            Spectrum type - 'Optimal' or 'Sum'
        showPlot: bool
            Show a plot with the spectrum?
        vmin: float or None
            Value minimum for dynamic spectrum image
        vmax: float or None
            Value maximum for dynamic spectrum image
        flipX: bool
            Flip the X axis?
        interpolation: None or str
            plot interpolation to use for matplotlib imshow()
        topYlabel: str
            The label for the top Y axis
        waveCal: bool
            Calibrate the dispersion to wavelength?
        smooth2D: None or 2 element list
            If None, no smoothing. If a 2 element list [x_std,y_std],
            astropy convolve with 2D Gaussian kernel
        """
        if os.path.exists(self.specFile) == False:
            raise Exception("No spectrum file found. Run extraction first...")
        
        HDUList = fits.open(self.specFile)
        if specType == 'Optimal':
            extSpec = HDUList['OPTIMAL SPEC'].data[src]
            errSpec = HDUList['OPT SPEC ERR'].data[src]
        elif specType == 'Sum':
            extSpec = HDUList['SUM SPEC'].data[src]
            errSpec = HDUList['SUM SPEC ERR'].data[src]
        elif specType == 'Background':
            extSpec = HDUList['BACKGROUND SPEC'].data[src]
            errSpec = HDUList['BACKGROUND SPEC'].data[src] * np.nan
        else:
            raise Exception("Unrecognized SpecType {}".format(specType))
        
        nImg = extSpec.shape[0]
        
        if extraFF == True:
            x, y, yerr = self.get_spec(src=src)
            cleanY = np.zeros_like(y)
            goodPts = np.isfinite(y)
            cleanY[goodPts] = y[goodPts]
            yFlat = utils.flatten(x,cleanY,highPassFreq=0.05)
            ySmooth = y - yFlat
            specFF = np.ones_like(y)
            dispPix = int(self.param['dispPixels'] + self.dispOffsets[src])
            
            disp1, disp2 = dispPix[0] + 10, dispPix[1] - 10
            specFF[disp1:disp2] = y[disp1:disp2] / ySmooth[disp1:disp2]
            badPt = np.isfinite(specFF) == False
            specFF[badPt] = 1.0
            specFF2D = np.tile(specFF,[nImg,1])
            extSpec = extSpec / specFF2D
        
        if align == True:
            useSpec, specOffsets = self.align_spec(extSpec,diagnostics=alignDiagnostics,
                                                   srcInd=src)
        else:
            useSpec = extSpec
            specOffsets = np.zeros(nImg)
        
        ## Expecting a RuntimeWarning if all NaN a long a slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avgSpec = np.nanmean(useSpec,0)
        
        specCounts = np.ones_like(errSpec)
        nanPt = (np.isfinite(useSpec) == False) | (np.isfinite(errSpec) == False)
        specCounts[nanPt] = np.nan
        
        ## Expecting a Runtime Warning if all NaN along a slice again
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avgSpec_err = np.sqrt(np.nansum(errSpec**2,0)) / np.nansum(specCounts,0)
        
        waveIndices = HDUList['DISP INDICES'].data
        normImg = np.tile(avgSpec,[nImg,1])
        dynamicSpec = useSpec / normImg
        dynamicSpec_err = errSpec / normImg
        
        if saveFits == True:
            dynHDU = fits.PrimaryHDU(dynamicSpec,HDUList['OPTIMAL SPEC'].header)
            dynHDU.name = 'DYNAMIC SPEC'
            dynHDU.header['ALIGNED'] = (align, 'Are the spectra shifted to align with each other?')
            dynHDU.header['SPECTYPE'] = (specType, 'Type of spectrum (sum versus optimal)')
            
            dynHDUerr = fits.ImageHDU(dynamicSpec_err,dynHDU.header)
            dynHDUerr.name = 'DYN SPEC ERR'
            dispHDU = HDUList['DISP INDICES']
            timeHDU = HDUList['TIME']
            
            ## the shift direction is opposite the measured shift, so save this
            offsetHDU = fits.ImageHDU(-specOffsets) 
            offsetHDU.name = 'SPEC OFFSETS'
            offsetHDU.header['BUNIT'] = ('pixels','units of Spectral offsets')
            
            avgHDU = fits.ImageHDU(avgSpec)
            avgHDU.name = 'AVG SPEC'
            
            avgErrHDU = fits.ImageHDU(avgSpec_err)
            avgErrHDU.name = 'AVG SPEC ERR'
            
            ## Save the wavelength array
            hduWave = fits.ImageHDU(self.wavecal(waveIndices))
            hduWave.header['AXIS1'] = ('wavelength','wavelength (microns)')
            hduWave.header['BUNIT'] = ('microns','wavelength unit')
            hduWave.name = 'WAVELENGTH'
            
            ## Expect a card-too-long warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message= "Card is too long, comment will be truncated")
                outHDUList = fits.HDUList([dynHDU,dynHDUerr,dispHDU,timeHDU,offsetHDU,avgHDU, avgErrHDU, hduWave])
                if align == True:
                    if specType == 'Optimal':
                        alignedHDU = fits.ImageHDU(useSpec,HDUList['OPTIMAL SPEC'].header)
                    else:
                        alignedHDU = fits.ImageHDU(useSpec,HDUList['SUM SPEC'].header)
                    ## Save the aligned spectra
                    outHDUList.append(alignedHDU)
                if specType != 'Background':
                    outHDUList.writeto(self.dyn_specFile(src),overwrite=True)
        
        if specAtTop == True:
            fig, axArr = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': [1, 3]})
            axTop = axArr[0]
            if waveCal == True:
                x_spec = self.wavecal(waveIndices)
            else:
                x_spec = waveIndices
            
            axTop.plot(x_spec,avgSpec)
            axTop.set_ylabel(topYlabel)
            ax = axArr[1]
        else:
            fig, ax = plt.subplots()
        
        if vmin is None:
            vmin=0.95
        if vmax is None:
            vmax=1.05
        
        if waveCal == True:
            all_waves = self.wavecal(waveIndices)
            extent = [all_waves[0],all_waves[-1],0,dynamicSpec.shape[0]]
        else:
            extent = None
        
        if smooth2D is None:
            zShow = dynamicSpec
        else:

            
            kernel = Gaussian2DKernel(x_stddev=smooth2D[0],y_stddev=smooth2D[1])
            if kernel.shape[0] * kernel.shape[1] > 15000:
                eTxt = "Large kernel sizes will hang. This one is {}x{}".format(kernel.shape[0],
                                                                                kernel.shape[1])
                raise Exception(eTxt)
            
            zShow = convolve_fft(dynamicSpec,kernel=kernel)
        
        imShowData = ax.imshow(zShow,vmin=vmin,vmax=vmax,origin='lower',
                               extent=extent,interpolation=interpolation,
                               rasterized=True)
        ax.set_aspect('auto')
        if waveCal == True:
            ax.set_xlabel(r'Wavelength ($\mu$m)')
        else:
            ax.set_xlabel('Disp (pixels)')
        
        ax.set_ylabel('Time (Image #)')
        dispPix = np.array(self.param['dispPixels']) + self.dispOffsets[src]
        if waveCal == True:
            disp_X = self.wavecal(dispPix)
        else:
            disp_X = dispPix
        
        if flipX == True:
            ax.set_xlim(disp_X[1],disp_X[0])
        else:
            ax.set_xlim(disp_X[0],disp_X[1])
        
        fig.colorbar(imShowData,label='Normalized Flux')
        if specAtTop == True:
            ## Fix the axes to be the same
            pos = ax.get_position()
            pos2 = axTop.get_position()
            axTop.set_position([pos.x0,pos2.y0,pos.width,pos2.height])
            
        
        dyn_spec_name = '{}_dyn_spec_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
        dyn_spec_path = os.path.join(self.baseDir,'plots','spectra','dynamic_spectra',dyn_spec_name)
        fig.savefig(dyn_spec_path,bbox_inches='tight',dpi=150)
        
        if showPlot == True:
            fig.show()
        else:
            plt.close(fig)
        
        HDUList.close()
    
    def plot_noise_spectrum(self,src=0,showPlot=True,yLim=None,
                            startInd=0,endInd=None,waveCal=False,
                            waveBin=False,nbins=10,
                            returnNoiseSpec=False):
        """
        Plot the Noise Spectrum from the Dynamic Spectrum
        
        Parameters
        ----------
        src: int
            Index number for the source
        showPlot: bool
            Show a plot in backend?
        yLim: list or None
            Specify the y limits
        startInd: int
            Starting time index to use
        endInd: int or None
            Ending index to use. None will use all
        
        waveCal: bool
            Wavelength calibrate the dispersion pixel
                            
        waveBin: bool
            Bin the wavelengths ?
        nbins: int
            How many wavelength bins should be used?
        
        returnNoiseSpec: bool
            Return the noise spectrum? If True, an astropy table is returned
        """
        fig, ax = plt.subplots()
        
        if waveBin == True:
            res = self.print_noise_wavebin(nbins=nbins,startInd=startInd,endInd=endInd)
            if waveCal == True:
                x_plot = res['Wave (mid)']
            else:
                x_plot = res['Disp Mid']
            y = res['Stdev (%)'] / 100.
            theo_y = res['Theo Err (%)'] / 100.
        else:
            HDUList = fits.open(self.dyn_specFile(src))
            x = HDUList['DISP INDICES'].data
            if waveCal == True:
                x_plot = self.wavecal(x)
            else:
                x_plot = x
            
            y = np.nanstd(HDUList['DYNAMIC SPEC'].data[startInd:endInd,:],axis=0)
            
            theo_y = np.nanmedian(HDUList['DYN SPEC ERR'].data[startInd:endInd,:],axis=0)
        
        if waveCal == True:
            xLabel = r'Wavelength ($\mu$m)'
        else:
            xLabel = "Disp Pixel"
        
        ax.plot(x_plot,y * 100.,label='Measured noise')
        ax.plot(x_plot,theo_y * 100.,label='Theoretical noise')
        ax.set_xlabel(xLabel)
        ax.set_ylabel("Noise (%)")
        if yLim is not None:
            ax.set_ylim(yLim[0],yLim[1])
        ax.legend()
        
        if showPlot == True:
            fig.show()
        else:
            outName = 'noise_spec_{}.pdf'.format(self.dataFileDescrip)
            outPath = os.path.join(self.baseDir,'plots','spectra','noise_spectrum',outName)
            print("Writing noise spectrum to {}".format(outPath))
            fig.savefig(outPath)
        if waveBin == False:
            HDUList.close()
        
        if returnNoiseSpec == True:
            t = Table()
            t['x'] = x_plot
            t['y'] = y
            t['theo_y'] = theo_y
            return t
    
    def plot_spec_offsets(self,src=0):
        if os.path.exists(self.dyn_specFile(src)) == False:
            self.plot_dynamic_spec(src=src,saveFits=True)
        HDUList = fits.open(self.dyn_specFile(src))
        time = HDUList['TIME'].data
        specOffset = HDUList['SPEC OFFSETS'].data
        fig, ax = plt.subplots()
        offset_time = self.get_offset_time(time)
        ax.plot(time - offset_time,specOffset)
        ax.set_xlabel("JD - {} (days)".format(offset_time))
        ax.set_ylabel("Dispersion Offset (px)")
        
        outName = 'spec_offsets_{}.pdf'.format(self.dataFileDescrip)
        outPath = os.path.join(self.baseDir,'plots','spectra','spec_offsets',outName)
        fig.savefig(outPath)
        plt.close(fig)
    
    def wavebin_specFile(self,nbins=10,srcInd=0):
        """
        The name of the wavelength-binned time series file
        
        Parameters
        -----------
        nbins: int
            The number of wavelength bins
        srcInd: int
            The index for the source. For a single source the index is 0.
        """
        return "{}_wavebin_{}_src_{}.fits".format(self.wavebin_file_prefix,nbins,srcInd)
    
    
    def align_dynamic_spectra(self,alignStars=True,starAlignDiagnostics=False,
                              skipIndividualDynamic=False,
                              **kwargs):
        """
        Align the dynamic spectra for multi-object spectroscopy?
        This method uses the mosOffsets to get the gross alignments of the 
        dynamic spectra.
        
        Parameters
        ----------
        alignStars: bool
            Cross-correlate to find the individual star's offsets?
        starAlignDiagnostics: bool
            Show the diagnostics for aligning average aspectra
        skipIndividualDynamic: bool
            Skip the individual dynamic spectra. Use this when trying to run
                quickly many times and use the saved versions only.
        """
        specHead = fits.getheader(self.specFile)
        nDispPixels = self.param['dispPixels'][1] - self.param['dispPixels'][0]
        combined_dyn = np.zeros([self.nsrc,specHead['NIMG'],nDispPixels])
        combined_dyn_err = np.zeros_like(combined_dyn)
        
        avgSpecs = np.zeros([self.nsrc,nDispPixels])
        for oneSrc in np.arange(self.nsrc):
            dispSt, dispEnd = np.array(np.array(self.param['dispPixels']) + self.dispOffsets[oneSrc],
                                       dtype=int)
            if skipIndividualDynamic == False:
                self.plot_dynamic_spec(src=oneSrc,**kwargs)
            HDUList = fits.open(self.dyn_specFile(oneSrc))
            combined_dyn[oneSrc,:,:] = HDUList['DYNAMIC SPEC'].data[:,dispSt:dispEnd]
            combined_dyn_err[oneSrc,:,:] = HDUList['DYN SPEC ERR'].data[:,dispSt:dispEnd]
            
            avgSpecs[oneSrc,:] = HDUList['AVG SPEC'].data[dispSt:dispEnd]
            
            #plt.plot(avgSpecs[oneSrc,:]/np.nanmax(avgSpecs[oneSrc,:]))
            if oneSrc == 0:
                outHDUList = fits.HDUList(HDUList)
                #outHDUList['DISP INDICES'].data = HDUList['DISP INDICES'].data[dispSt:dispEnd]
        
        if alignStars == True:
            refSpec = avgSpecs[self.nsrc // 2,:]
            waveIndices = np.arange(dispEnd - dispSt)
            Noffset = self.param['nOffsetCC']
            starOffsets = np.zeros(self.nsrc)
            for oneSrc in np.arange(self.nsrc):
                thisSpec = avgSpecs[oneSrc,:]
                    
                offsetX, offsetInd = utils.crosscor_offset(waveIndices,refSpec,thisSpec,Noffset=Noffset,
                                                           diagnostics=starAlignDiagnostics,
                                                           subPixel=True,
                                                           lowPassFreq=self.param['lowPassFreqCC'],
                                                           highPassFreq=self.param['hiPassFreqCC'])
                if starAlignDiagnostics == True:
                    pdb.set_trace()
                
                for oneImg in np.arange(specHead['NIMG']):
                    tmp = utils.roll_pad(combined_dyn[oneSrc,oneImg,:],
                                         offsetInd * self.param['specShiftMultiplier'])
                    combined_dyn[oneSrc,oneImg,:] = tmp
                starOffsets[oneSrc] = offsetInd * self.param['specShiftMultiplier']            
        else:
            starOffsets = np.zeros_like(self.nsrc)
        
        
        outHDUList['DYNAMIC SPEC'].data = combined_dyn
        outHDUList['DYN SPEC ERR'].data = combined_dyn_err
        
        print("Output written to {}".format(self.dyn_specFile('comb')))
        outHDUList.writeto(self.dyn_specFile('comb'),overwrite=True)
        HDUList.close()
        
        #plt.show()
    
    def refcorrect_dynamic_spectra(self,mainSrcIndex=0):
        """
        Divide the (aligned) dynamic spectra of one source by another for MOS
        
        Parameters
        ----------
        mainSrcIndex: int
            The main source that will be divided by references
        """
        HDUList = fits.open(self.dyn_specFile('comb'))
        outHDU = fits.HDUList(HDUList)
        head = HDUList['DYNAMIC SPEC'].header
        dyn_specRatio = np.zeros([self.nsrc,head['NIMG'],head['NAXIS1']])
        try:
            mainSrc = HDUList['DYNAMIC SPEC'].data[mainSrcIndex]
            for srcInd in np.arange(self.nsrc):
                dyn_specRatio[srcInd] = mainSrc / HDUList['DYNAMIC SPEC'].data[srcInd]
        except RuntimeWarning:
             pass
        
        outHDU[0].data = dyn_specRatio
        outHDU[0].name = 'DYNAMIC SPECTRATIO'
        print("writing output to {}".format(self.dyn_specFile('ratio')))
        outHDU.writeto(self.dyn_specFile('ratio'),overwrite=True)
    
    def make_wavebin_series(self,specType='Optimal',src=0,nbins=10,dispIndices=None,
                            recalculate=False,align=False,refCorrect=False,
                            binStarts=None,binEnds=None):
        """
        Bin wavelengths together and generate a time series from the dynamic spectrum
        
        Parameters
        ----------
        specType: str
            Type of extraction 'Optimal' vs 'Sum'
        src: int
            Which source index is this for
        nbins: int
            Number of wavelength bins
        dispIndices: list of ints
            The detector pixel indices over which to create the wavelength bins
        recalculate: bool
            Re-caalculate the dynamic spectrum?
        align: bool
            Automatically align all the spectra? This is passed to plot_dynamic_spec
        refCorrect: bool
            Use the reference corrected photometry?
        binStarts: numpy array or None
            Specify starting points for the bins. If None, binStarts are calculated automatically
        binEnds: numpy array or None
            Specify starting points for the bins w/ Python, so the last point is binEnds - 1.
            If None, binEnds are calculated automatically.
        """
        
        ## Check if there is a previous dynamic spec file with same parameters
        
        if (os.path.exists(self.dyn_specFile(src)) == True):
            head_dyn = fits.getheader(self.dyn_specFile(src),ext=0)
            if phot_pipeline.exists_and_equal(head_dyn,'ALIGNED',align):
                previousFile = False
            elif phot_pipeline.exists_and_equal(head_dyn,'SPECTYPE',specType):
                previousFile = False
            else:
                previousFile = True
        else:
            previousFile = False
        
        if (previousFile == False) | (recalculate == True):
            print('Remaking dynamic spectrum...')
            self.plot_dynamic_spec(src=src,saveFits=True,align=align,specType=specType)
        
        if refCorrect == True:
            HDUList = fits.open(self.dyn_specFile('ratio'))
            dynSpec = HDUList['DYNAMIC SPECTRATIO'].data[src]
        
            dynSpec_err = HDUList['DYN SPEC ERR'].data[src]
        else:
            HDUList = fits.open(self.dyn_specFile(src))
            dynSpec = HDUList['DYNAMIC SPEC'].data
        
            dynSpec_err = HDUList['DYN SPEC ERR'].data
        goodp = np.isfinite(dynSpec_err) & (dynSpec_err != 0)
        
        time = HDUList['TIME'].data
        nTime = len(time)
        
        ## Time variable weights
        weights_tvar = np.zeros_like(dynSpec_err)
        weights_tvar[goodp] = 1./dynSpec_err[goodp]**2
        ## Weights that are time-constant
        meanWeight = np.nanmean(weights_tvar,0)
        weights = np.tile(meanWeight,[nTime,1])
        
        disp = HDUList['DISP INDICES'].data
        
        if dispIndices == None:
            dispSt, dispEnd = np.array(np.array(self.param['dispPixels']) + self.dispOffsets[src],dtype=int)
        else:
            dispSt, dispEnd = dispIndices
        
        binEdges = np.array(np.linspace(dispSt,dispEnd,nbins+1),dtype=int)
        if binStarts is None:
            binStarts = binEdges[0:-1]
        if binEnds is None:
            binEnds = binEdges[1:]
        binIndices = np.arange(len(binStarts))
        
        binGrid = np.zeros([nTime,nbins])
        binGrid_err = np.zeros_like(binGrid)
        
        binned_disp = np.zeros(nbins)
        
        
        for ind, binStart, binEnd in zip(binIndices,binStarts,binEnds):
            theseWeights = weights[:,binStart:binEnd]
            binGrid[:,ind] = np.nansum(dynSpec[:,binStart:binEnd] * theseWeights ,1)
            normFactor = np.nanmedian(binGrid[:,ind])
            binGrid[:,ind] = binGrid[:,ind] / normFactor
            binGrid_err[:,ind] = np.sqrt(np.nansum((dynSpec_err[:,binStart:binEnd] * theseWeights)**2,1))
            binGrid_err[:,ind] = binGrid_err[:,ind] / normFactor
            
            binned_disp[ind] = np.mean([binStart,binEnd])
            
            #plt.errorbar(time - offset_time,binGrid[:,ind] - 0.005 * ind,fmt='o')
            #plt.errorbar(time - offset_time,binGrid[:,ind] - 0.005 * ind,
            #             yerr=binGrid_err[:,ind])
        
        outHDU = fits.PrimaryHDU(binGrid,HDUList[0].header)
        outHDU.name = 'BINNED F'
        errHDU = fits.ImageHDU(binGrid_err,outHDU.header)
        errHDU.name = 'BINNED ERR'
        offsetHDU = HDUList['SPEC OFFSETS']
        timeHDU = HDUList['TIME']
        
        dispTable = Table()
        dispTable['Bin Start'] = binStarts
        dispTable['Bin Middle'] = binned_disp
        dispTable['Bin End'] = binEnds
        
        dispHDU = fits.BinTableHDU(dispTable)
        dispHDU.name = "DISP INDICES"
        
        waveTable = Table()
        for oneColumn in dispTable.colnames:
            waveTable[oneColumn] = self.wavecal(dispTable[oneColumn])
        waveHDU = fits.BinTableHDU(waveTable)
        waveHDU.name = 'WAVELENGTHS'
        
        outHDUList = fits.HDUList([outHDU,errHDU,timeHDU,offsetHDU,dispHDU,waveHDU])
        outHDUList.writeto(self.wavebin_specFile(nbins,src),overwrite=True)
        
        HDUList.close()
    
    def get_offset_time(self,time):
        return np.floor(np.min(time))
    
    def plot_wavebin_series(self,nbins=10,offset=0.005,showPlot=False,yLim=None,xLim=None,
                            recalculate=False,dispIndices=None,differential=False,
                            interactive=False,unit='fraction',align=False,specType='Optimal',
                            src=0,refCorrect=False,binStarts=None,binEnds=None,
                            timeBin=False,nTimeBin=150,waveLabels=False):
        """
        Plot a normalized lightcurve for wavelength-binned data one wavelength at a time with
        an offset between the lightcurves.
        
        Parameters
        ----------
        nbins: int
            The number of wavelength bins to use
        offset: float
            The normalized flux offset between wavelengths
        showPlot: bool
            Show the plot widget?
            The result also depends on the interactive keyword. If interactive is True,
            showPlot is ignored and it saves
            an html file in plots/spectra/interactive/.
            If False (and interactive is False), it is saved in 
            plots/spectra/wavebin_tseries/.
            If True (and interactive is False),
            no plot is saved and instead the plot is displayed in the
            matplotlib framework.
        yLim: list of 2 elements or None
            If None, it will automatically set the Y axis. If a list [y_min,y_max],
            it will make the Y axis limits y_min to y_max
        xLim: list of 2 elements or None
            If None, it will automatically set the X axis. If a list [x_min,x_max],
            it will make the X axis limits x_min to x_max
        recalculate: bool
            If True, it will calculate the values from the dynamic spectrum.
            If False, it will used cached results from a previous set of lightcurves.
        dispIndices: list of 2 elements or None
            If None, it will use the dispIndices from the parameter file.
            If a 2 element list of [disp_min,disp_max], it will use those coordinates
        differential: bool
            If True, it will first divide each wavelength's lightcurve by the lightcurve average
            of all wavelengths. This means that it shows the differential time series from average.
            If False, no division is applied (other than normalization)
        interactive: bool
            If True, it will use :code:`bokeh` to create an interactive HTML javascript plot
            where you can hover over data points to find out the file name for them. It saves
            the html plot in plots/spectra/interactive/. If False,
            it will create a regular matplotlib plot.
        unit: str
            Flux unit. 'fraction' or 'ppm'
        specType: str
            Type of extraction 'Optimal' vs 'Sum'
        align: bool
            Automatically align all the spectra? This is passed to plot_dynamic_spec
        src: int
            Index number for which spectrum to look at (used for Multi-object spectroscopy)
        refCorrect: bool
            Correct by reference stars for MOS?
        timeBin: bool
            Bin the points in time?
        nTimeBin
            Number of points to bin in time
        """
        if (os.path.exists(self.wavebin_specFile(nbins=nbins,srcInd=src)) == False) | (recalculate == True):
            self.make_wavebin_series(nbins=nbins,dispIndices=dispIndices,recalculate=recalculate,
                                     specType=specType,align=align,src=src,refCorrect=refCorrect,
                                     binStarts=binStarts,binEnds=binEnds)
        
        HDUList = fits.open(self.wavebin_specFile(nbins=nbins,srcInd=src))
        time = HDUList['TIME'].data
        offset_time = self.get_offset_time(time)
        
        disp = HDUList['DISP INDICES'].data
        
        binGrid = HDUList['BINNED F'].data
        binGrid_err = HDUList['BINNED ERR'].data
        
        if waveLabels == True:
            waves_table = Table(HDUList['WAVELENGTHS'].data)
            waves = np.array(waves_table['Bin Middle'])
        
        if differential == True:
            weights = 1./(np.nanmean(binGrid_err,0))**2
            weights2D = np.tile(weights,[binGrid.shape[0],1])
            
            avgSeries = (np.nansum(binGrid * weights2D,1)) / np.nansum(weights2D,1)
            
            avgSeries2D = np.tile(avgSeries,[len(disp),1]).transpose()
            binGrid = binGrid / avgSeries2D
        
        if interactive == True:
            
            outFile = "refseries_{}.html".format(self.dataFileDescrip)
            outPath = os.path.join(self.baseDir,'plots','spectra','interactive',outFile)
            
            fileBaseNames = []
            fileTable = Table.read(self.specFile,hdu='FILENAMES')
            indexArr = np.arange(len(fileTable))
            for oneFile in fileTable['File Path']:
                fileBaseNames.append(os.path.basename(oneFile))
            
            bokeh.plotting.output_file(outPath)
            
            dataDict = {'t': time - offset_time,'name':fileBaseNames,'ind':indexArr}
            for ind,oneDisp in enumerate(disp):
                dataDict["y{:02d}".format(ind)] = binGrid[:,ind] - offset * ind
            
            source = ColumnDataSource(data=dataDict)
            p = bokeh.plotting.figure()
            p.background_fill_color="#f5f5f5"
            p.grid.grid_line_color="white"
            p.xaxis.axis_label = 'Time (JD - {})'.format(offset_time)
            p.yaxis.axis_label = 'Normalized Flux'
            
            colors = itertools.cycle(palette)
            for ind,oneDisp in enumerate(disp):
                p.circle(x='t',y="y{:02d}".format(ind),source=source,color=next(colors),
                         size=5)
            
            p.add_tools(HoverTool(tooltips=[('name', '@name'),('index','@ind')]))
            bokeh.plotting.show(p)
            
        else:
            fig, ax = plt.subplots()
            for ind,oneDisp in enumerate(disp):
                if unit == 'fraction':
                    y_plot_full_tres = binGrid[:,ind] - offset * ind
                else:
                    y_plot_full_tres = (binGrid[:,ind] - 1.0) * 1e6 - offset * ind * 1e6
                
                if timeBin == True:
                    goodP = np.ones_like(time,dtype=bool)
                    y_plot, xEdges, binNum = binned_statistic(time[goodP],y_plot_full_tres[goodP],
                                                              bins=nTimeBin)
                    time_plot = (xEdges[:-1] + xEdges[1:])/2.
                else:
                    y_plot = y_plot_full_tres
                    time_plot = time
                
                ax.errorbar(time_plot - offset_time,y_plot,fmt='o')
                
                if waveLabels == True:
                    ax.text(time_plot[-1] - offset_time,y_plot[-1],r'{:.2f} $\mu$m'.format(waves[ind]))
            
            if yLim is not None:
                ax.set_ylim(yLim[0],yLim[1])
            
            if xLim is not None:
                ax.set_xlim(xLim[0],xLim[1])
            
            ax.set_xlabel('Time (JD - {})'.format(offset_time))
            if unit == 'fraction':
                ax.set_ylabel('Normalized Flux')
            else:
                ax.set_ylabel('(Normalized Flux - 1.0) (ppm)')
            
            if showPlot == True:
                fig.show()
            else:
                outName = 'wavebin_tser_{}_src_{}.pdf'.format(self.dataFileDescrip,src)
                outPath = os.path.join(self.baseDir,'plots','spectra','wavebin_tseries',outName)
                fig.savefig(outPath,bbox_inches='tight')
                plt.close(fig)

            
        HDUList.close()
    
    
    def get_wavebin_series(self,nbins=10,recalculate=False,specType='Optimal',srcInd=0,
                           binStarts=None,binEnds=None):
        """
        Get a table of the the wavelength-binned time series
        
        
        Parameters
        -----------
        nbins: int
            The number of wavelength bins
        
        recalculate: bool, optional
            Recalculate the dynamic spectrum?
            This is good to set as True when there is an update to
            the parameter file.
        
        specType: str, optional
            The type of spectral extraction routine
            Eg. "Sum" for sum extraction, "Optimal" for optimal extraction
            This will be skipped over if recalculate=False and a file already exists
        
        srcInd: int, optional
            The index of the source. For single objects it is 0.
        
        binStarts: int, optional or None
            Pixel starting positions (or None). If None, it will be calculated as a linear spacing
        
        binEnds: int, optional
            Pixel ending positions (or None). If None, it will be calculated as a linear spacing
        
        Returns
        --------
        t1: astropy table
            A table of wavelength-binned flux values
        t2: astropy table
            A table of wavelength-binned error values
        
        Examples
        --------
        
        >>> from tshirt.pipeline import spec_pipeline
        >>> spec = spec_pipeline.spec()
        >>> t1, t2 = spec.get_wavebin_series()
        """
        sFile = self.wavebin_specFile(nbins=nbins,srcInd=srcInd)
        if (os.path.exists(sFile) == False) | (recalculate == True):
            self.plot_wavebin_series(nbins=nbins,recalculate=recalculate,specType=specType,
                                     binStarts=binStarts,binEnds=binEnds)
        HDUList = fits.open(sFile)
        disp = HDUList['DISP INDICES'].data
        binGrid = HDUList['BINNED F'].data
        binGrid_err = HDUList['BINNED ERR'].data
        time = HDUList['TIME'].data
        t1, t2 = Table(), Table()
        t1['Time'] = time
        t2['Time'] = time
        
        for ind,oneBin in enumerate(disp['Bin Middle']):
            wave = np.round(self.wavecal(oneBin),3)
            t1['{:.3f}um Flux'.format(wave)] = binGrid[:,ind]
            t2['{:.3f}um Error'.format(wave)] = binGrid_err[:,ind]
        
        HDUList.close()
        return t1, t2
    
    def print_noise_wavebin(self,nbins=10,shorten=False,recalculate=False,align=False,
                            specType='Optimal',npoints=15,srcInd=0,
                            startInd=0,endInd=None):
        """ 
        Get a table of noise measurements for all wavelength bins
        
        Parameters
        ----------
        nbins: int
            The number of wavelength bins
        shorten: bool
            Use a short segment of the full time series?
            This could be useful for avoiding bad data or a deep transit.
            This overrides startInd and endInd so choose either shorten or specify
            the start and end indices
        npoints: int
            Number of points to include in the calculation (only if shorten is True)
            This will only use the first npoints.
        startInd: int
            Starting time index to use (for specifying a time interval via indices)
            Shorten will supercede this so keep it False to use startInd.
        endInd: int
            Ending time index to use (for specifying a time interval via indices).
            Shorten will supercede this so keep it False to use endInd.
        recalculate: bool
            Recalculate the wavebin series and dynamic spectrum?
        specType: str
            Type of extraction 'Optimal' vs 'Sum'
        align: bool
            Automatically align all the spectra? This is passed to plot_dynamic_spec
        srcInd: int
            Index for the source. For a single object, the index is 0.
        
        Returns
        ---------
        t: an astropy.table object
            A table of wavelength bins, with theoretical noise
            and measured standard deviation across time
        """
        sFile = self.wavebin_specFile(nbins=nbins,srcInd=srcInd)
        if (os.path.exists(sFile) == False) | (recalculate==True):
            self.plot_wavebin_series(nbins=nbins,recalculate=recalculate,specType=specType,
                                     align=align)
        HDUList = fits.open(sFile)
        disp = HDUList['DISP INDICES'].data
        binGrid = HDUList['BINNED F'].data
        binGrid_err = HDUList['BINNED ERR'].data
        
        t = Table()
        t['Disp St'] = disp['Bin Start']
        t['Disp Mid'] = disp['Bin Middle']
        t['Disp End'] = disp['Bin End']
        waveStart = self.wavecal(disp['Bin Start'])
        waveEnd = self.wavecal(disp['Bin End'])
        t['Wave (st)'] = np.round(waveStart,3)
        t['Wave (mid px)'] = np.round(self.wavecal(t['Disp Mid']),3)
        t['Wave (mid)'] = np.round((waveStart + waveEnd)/2.,3)
        t['Wave (end)'] = np.round(waveEnd,3)
        if shorten == True:
            binGrid = binGrid[0:npoints,:]
        else:
            binGrid = binGrid[startInd:endInd,:]
        t['Stdev (%)'] = np.round(np.std(binGrid,axis=0) * 100.,4)
        t['Theo Err (%)'] = np.round(np.median(binGrid_err,axis=0) * 100.,4)
        
        medianV = np.median(binGrid,axis=0)
        absDeviation = np.abs(binGrid - medianV)
        t['MAD (%)'] = np.round(np.median(absDeviation,axis=0) * 100.,4)
        
        HDUList.close()
        return t
    
    def get_broadband_series(self,src=0,recalculate=False,**kwargs):
        """
        Return a table from the broadband time series

        Parameters
        ----------
        src: int
            Source index
        recalculate: bool
            Recalculate the time series?
        """
        nbins=1
        if ((os.path.exists(self.wavebin_specFile(nbins=nbins,srcInd=src)) == False) | 
            (recalculate == True)):
            self.make_wavebin_series(nbins=nbins,**kwargs)
        
        HDUList = fits.open(self.wavebin_specFile(nbins=nbins,srcInd=src))

        t = Table()
        t['Time'] = HDUList['TIME'].data
        
        binGrid = HDUList['BINNED F'].data
        binGrid_err = HDUList['BINNED ERR'].data
        
        #spec2D = HDUList['OPTIMAL SPEC'].data[src]
        #spec2D_err = HDUList['OPT SPEC ERR'].data[src]
        t['Flux'] = binGrid[:,0]
        t['Flux Err'] = binGrid_err[:,0]
        
        HDUList.close()
        
        return t
    
    def save_broadband_series(self,src=0,saveDir=None,
                              **kwargs):
        """
        Save the broadband series to file

        Parameters
        ----------
        src: int
            Source index
        saveDir: none or str
            Directory to save the file to. If None,
            it will go in tshirt/tser_data/broadbad_tseries
        kwargs: additional parameter to pass to 
            get_broadband_series and make_wavebin_series
        """
        t = self.get_broadband_series(src=src,**kwargs)
        if saveDir is None:
            saveDir = os.path.join(self.baseDir,'tser_data',
                                   'broadband_tseries')
        saveName = 'bb_'+self.dataFileDescrip +'.csv'
        savePath = os.path.join(saveDir,saveName)
        print("Saving broadband Time Series to {}".format(savePath))
        t.write(savePath,overwrite=True)

    def plot_broadband_series(self,src=0,showPlot=True,matchIraf=False):
        t = self.get_broadband_series(src=src)
        offset_time = self.get_offset_time(t['Time'])
        
        err_ppm = np.nanmedian(t['Flux Err']) * 1e6
        print('Formal Err = {} ppm '.format(err_ppm))
        fig, ax = plt.subplots()
        
        if matchIraf == True:
            marker = '.'
            x = np.arange(len(t))
            ax.set_xlabel("Int Number")
        else:
            marker = 'o'
            x = t['Time'] - offset_time
            ax.set_xlabel("Time - {} (days)".format(offset_time))
            
        
        ax.plot(x,t['Flux'],marker)
        ax.set_ylabel("Normalized Flux")
        if showPlot == True:
            fig.show()
        else:
            bb_series_name = '{}_bb_series_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
            outName = os.path.join(self.baseDir,'plots','spectra','broadband_series',bb_series_name)
            fig.savefig(outName)

    def calculate_apertures(self,src=0):
        """
        Calculate the source aperture (where to start/end)
        """
        startInd, endInd = self.param['dispPixels']
        dispersion_px = np.arange(startInd,endInd)
        spatialMid = self.eval_trace(dispersion_px,src=src)
        spatialMid_int = np.array(np.round(spatialMid),dtype=int)
        srcRadius_int = int(np.round(self.param['apWidth'] / 2.))
        srcStart = spatialMid_int - srcRadius_int
        srcEnd = spatialMid_int + srcRadius_int

        ## check that aperture is not outside boundaries of image
        img, head = self.get_default_im()
        if self.param['dispDirection'] == 'x':
            maxSpatial = img.shape[0] - 1
        else:
            maxSpatial = img.shape[1] - 1
        


        t = Table()
        t['dispersion_px'] = dispersion_px
        t['srcStart'] = np.maximum(srcStart,0)
        t['srcEnd'] = np.minimum(srcEnd,maxSpatial)
        t['spatialMid'] = spatialMid
        t['spatialMid_int'] = spatialMid_int

        if self.param['traceCurvedSpectrum'] == True:
            if self.param['dispDirection'] == 'x':
                bkgRegions = self.param['bkgRegionsY'][0]
            else:
                bkgRegions = self.param['bkgRegionsX'][0]
            bkgStarts0, bkgEnds1 = [], []
            
            t['bkgStart 0'] = int(bkgRegions[0])
            backInnerRadius_int = int(np.round(self.param['backgMinRadius']))
            t['bkgEnd 0'] = np.maximum(bkgRegions[0],t['spatialMid_int'] - backInnerRadius_int)
            t['bkgStart 1'] = np.minimum(bkgRegions[1],t['spatialMid_int'] + backInnerRadius_int)
            
            t['bkgEnd 1'] = int(bkgRegions[1])
        
        return t

    
    def showStarChoices(self,img=None,head=None,showBack=True,
                        srcLabel=None,figSize=None,vmin=None,vmax=None,
                        xlim=None,ylim=None,showPlot=False):
        """ Show the star choices for spectroscopy
                        
        Parameters
        ------------------
        img: numpy 2D array, optional
            An image to plot
        head: astropy FITS header, optional
            header for image
        showBack: bool, optional
            Show the background subtraction regions?
        srcLabel: str or None, optional
            What should the source label be?
            The default is "src"
        vmin: float, optional
            Minimum value for imshow
        vmax: float
            (optional) Maximum value for imshow
        figSize: 2 element list
            (optional) Specify the size of the plot
            This is useful for looking at high/lower resolution
        xlim: list or None
            (optional) Set the x limit of plot
        ylim: list or None
            (optional) Set the y limit of plot
        showPlot: bool
            Make the plot visible?
        """
        fig, ax = plt.subplots(figsize=figSize)
        
        img, head = self.get_default_im(img=img,head=None)
        
        if vmin == None:
            vmin = np.nanpercentile(img,1)
        
        if vmax == None:
            vmax = np.nanpercentile(img,99)
        
        imData = ax.imshow(img,cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.invert_yaxis()
        rad, txtOffset = 50, 20
        
        showApPos = self.param['starPositions']

        apWidth = self.param['apWidth']
        
        outName = 'spec_aps_{}.pdf'.format(self.dataFileDescrip)
        
        for ind, onePos in enumerate(showApPos):
            dispPos = np.array(self.param['dispPixels']) + self.dispOffsets[ind]
            dispLength = dispPos[1] - dispPos[0]
            
            if self.param['traceCurvedSpectrum'] == True:
                t = self.calculate_apertures(src=ind)
                dispersion_px, spatialMid = t['dispersion_px'], t['spatialMid']
                srcStart, srcEnd = t['srcStart'], t['srcEnd']
                if self.param['dispDirection'] == 'x':
                    ax.step(dispersion_px,srcStart,color='r')
                    ax.step(dispersion_px,srcEnd,color='r')
                    xPos = dispPos[0]
                    yPos = spatialMid[0]
                else:
                    ax.step(srcStart,dispersion_px,color='r')
                    ax.step(srcEnd,dispersion_px,color='r')
                    xPos = spatialMid[0]
                    yPos = dispPos[0]
                
            else:
                if self.param['dispDirection'] == 'x':
                    xPos = dispPos[0]
                    yPos = onePos - apWidth / 2.
                    width = dispLength
                    height = apWidth
                else:
                    xPos = onePos - apWidth / 2.
                    yPos = dispPos[0]
                    width = apWidth
                    height = dispLength
            
            
                rec = mpl.patches.Rectangle((xPos,yPos),width=width,height=height,fill=False,edgecolor='r')
                ax.add_patch(rec)
            
            if ind == 0:
                if srcLabel is None:
                    name='src'
                else:
                    name=srcLabel
            else:
                name=str(ind)
            
            if xlim is not None:
                xPos_show = np.max([xlim[0],xPos])
            else:
                xPos_show = xPos

            if ylim is not None:
                yPos_show = np.max([ylim[0],yPos])
            else:
                yPos_show = yPos

            ax.text(xPos_show+txtOffset,yPos_show+txtOffset,name,color='white')
        
        if showBack == True:
            if self.param['traceCurvedSpectrum'] == True:
                for ind in np.arange(self.nsrc):
                    t = self.calculate_apertures(src=ind)
                    for bkgCounter in ['0','1']:
                        arg1 = t['dispersion_px']
                        arg2 = t['bkgStart {}'.format(bkgCounter)]
                        arg3 = t['bkgEnd {}'.format(bkgCounter)]
                        bkgColor = 'orange'
                        bkgAlpha= 0.3
                        step='pre'
                        if self.param['dispDirection'] == 'x':
                            ax.fill_between(arg1,arg2,arg3,color=bkgColor,alpha=bkgAlpha,step=step)
                        else:
                            ax.fill_betweenx(arg1,arg2,arg3,color=bkgColor,alpha=bkgAlpha,step=step)
            else:
                for oneDirection in self.param['bkgSubDirections']:
                    if oneDirection == 'X':
                        bkPixels = self.param['bkgRegionsX']
                    elif oneDirection == 'Y':
                        bkPixels = self.param['bkgRegionsY']
                    else:
                        raise Exception("No Background subtraction direction {}".format(oneDirection))
                
                    for oneReg in bkPixels:
                        if oneDirection == 'X':
                            height = img.shape[0]
                            width = oneReg[1] - oneReg[0]
                            xPos = oneReg[0]
                            yPos = 0
                        else:
                            height = oneReg[1] - oneReg[0]
                            width = img.shape[1]
                            xPos = 0
                            yPos = oneReg[0]
                        rec = mpl.patches.Rectangle((xPos,yPos),width=width,height=height,color='orange',
                                                    alpha=0.3)
                        ax.add_patch(rec)
            
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        
        if xlim is not None:
            ax.set_xlim(xlim[0],xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0],ylim[1])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(imData,label='Counts',cax=cax)
        
        outPath = os.path.join(self.baseDir,'plots','spectra','star_labels',outName)
        fig.savefig(outPath, bbox_inches='tight')
        if showPlot == True:
            fig.show()
        else:
            plt.close(fig)
    
    def adjacent_bin_ratio(self,nbins=10,bin1=2,bin2=3,binMin=10,binMax=250,srcInd=0):
        """
        Examine the time series for adjacent bins
        
        Parameters
        ----------
        nbins: int
            Number of bins
        bin1: int
            Index of the first bin
        bin2: int
            Index of the second bin
        binMin: int
            start bin for allan variance
        binMax: int
            end bin for allan variance
        srcInd: int
            Index for the source. For a single source, the index is 0.
        """
        HDUList = fits.open(self.wavebin_specFile(nbins=nbins,srcInd=srcInd))
        
        time = HDUList['TIME'].data
        offset_time = self.get_offset_time(time)
        
        dat2D = HDUList['BINNED F'].data
        dat2Derr = HDUList['BINNED ERR'].data
        ratioSeries = dat2D[:,bin2] / dat2D[:,bin1]
        fracErr = np.sqrt((dat2Derr[:,bin2]/dat2D[:,bin2])**2 + (dat2Derr[:,bin1]/dat2D[:,bin1])**2)
        yerr = ratioSeries * fracErr
        
        stdRatio = np.std(ratioSeries) * 1e6
        print("stdev = {} ppm".format(stdRatio))
        
        phot_pipeline.allan_variance(time * 24. * 60.,ratioSeries * 1e6,yerr=yerr * 1e6,
                                     xUnit='min',yUnit='ppm',
                                     binMin=binMin,binMax=binMax)
        
        plt.plot(time - offset_time,ratioSeries)
        
        
        
        plt.show()
        HDUList.close()
        
    
    def wavecal(self,dispIndicesInput,waveCalMethod=None,head=None,
                dispShift=None,
                **kwargs):
        """
        Wavelength calibration to turn the dispersion pixels into wavelengths
        
        Parameters
        -------------
        dispIndices: numpy array 
            Dispersion Middle X-axis values
            
        waveCalMethod: str, optional 
            Corresponds to instrumentation used.
            Use 'NIRCamTS' for the NIRCam time series mode (:any:`jwst_inst_funcs.ts_wavecal`).
            Use 'wfc3Dispersion' for the HST WFC3 grism (:any:`ts_wavecal`).
            
        head: astropy FITS header, optional
            header for image
        
        dispShift: None or float
            Value by which to shift the disp indices before evaluating the wavelengths
            (for example a star is shifted). If None, it will use the value from the 
            parameter 'waveCalPxOffset'
        """
        if dispShift is None:
            dispOffsetPx = self.param['waveCalPxOffset']
        else:
            dispOffsetPx = dispShift

        dispIndices = np.array(dispIndicesInput) - dispOffsetPx

        if waveCalMethod == None:
            waveCalMethod = self.param['waveCalMethod']
        
        if waveCalMethod == None:
            wavelengths = dispIndices
        elif waveCalMethod == 'NIRCamTS':
            if head == None:
                head = fits.getheader(self.specFile,extname='ORIG HEADER')
            wavelengths = instrument_specific.jwst_inst_funcs.ts_wavecal(dispIndices,obsFilter=head['FILTER'],
                                                                         **kwargs)
        elif waveCalMethod == 'NIRCamTSquickPoly':
            if head == None:
                head = fits.getheader(self.specFile,extname='ORIG HEADER')
            wavelengths = instrument_specific.jwst_inst_funcs.ts_wavecal_quick_nonlin(dispIndices,obsFilter=head['FILTER'],
                                                                         **kwargs)
        elif waveCalMethod == 'simGRISMC':
            wavelengths = instrument_specific.jwst_inst_funcs.ts_grismc_sim(dispIndices,**kwargs)
        
        elif waveCalMethod == 'wfc3Dispersion':
            wavelengths = instrument_specific.hst_inst_funcs.hstwfc3_wavecal(dispIndices,**kwargs)
        elif waveCalMethod == 'wfc3G256':
            wavelengths = instrument_specific.hst_inst_funcs.hst_wavecal_grism256(dispIndices)
        elif waveCalMethod == 'quick_nrs_prism':
            wavelengths = instrument_specific.jwst_inst_funcs.quick_nirspec_prism(dispIndices)
        elif waveCalMethod == 'nrs_grating':
            if head == None:
                head = fits.getheader(self.specFile,extname='ORIG HEADER')
            wavelengths = instrument_specific.jwst_inst_funcs.nirspec_grating(dispIndices,head)

        elif waveCalMethod == 'grismr_poly_dms':
            if head == None:
                head = fits.getheader(self.specFile,extname='ORIG HEADER')
                obsFilter = head['FILTER']
            else:
                if 'FILTER' in head:
                    obsFilter = head['FILTER']
                else:
                    warning.warn('No filter specified. Assuming F322W2')
                    obsFilter = 'F322W2'
            wavelengths = instrument_specific.jwst_inst_funcs.flight_poly_grismr_nc(dispIndices,obsFilter=obsFilter)
        elif waveCalMethod == 'miri_lrs':
            wavelengths = instrument_specific.jwst_inst_funcs.miri_lrs(dispIndices)
        else:
            raise Exception("Unrecognized wavelength calibration method {}".format(waveCalMethod))
            
        wavelengths = wavelengths - self.param['waveCalOffset']
        return wavelengths
        
    def inverse_wavecal(self,waveArr):
        """
        Calculate the pixel location for a given wavelength
        Uses 1D interpolation of wavelength vs pixel
        """
        waveArr_use = np.array(waveArr)
        xArray = np.arange(self.param['dispPixels'][0]-4,
                           self.param['dispPixels'][1]+4,1)
        interp_fun = interp1d(self.wavecal(xArray),xArray)
        px_out = interp_fun(waveArr)
        return px_out
    
    def find_px_bins_from_waves(self,waveMid,waveWidth):
        """
        Given a set of wavelengths centers and widths, find the pixels
        that will approximately give you those wavelengths, but with 
        no pixels repeated or skipped.
        """
        waveEdges = np.array(waveMid) - np.array(waveWidth) * 0.5
        waveEdges = np.append(waveEdges,waveMid[-1] + waveWidth[-1] * 0.5) ## last edge
        pxEdges = np.array(np.round(self.inverse_wavecal(waveEdges)),dtype=int)

        t = Table()
        t['px start'] = pxEdges[0:-1]
        t['px end'] = pxEdges[1:]
        t['wave start'] = self.wavecal(t['px start'])
        t['wave end'] = self.wavecal(t['px end'])
        t['px mid'] = (t['px start'] + t['px end'])/2.
        t['px width'] = t['px end'] - t['px start']
        t['wave mid'] = (t['wave start'] + t['wave end'])/2.
        t['wave width'] = t['wave end'] - t['wave start']

        ## Check whether dispersion is negative
        
        if np.sum(t['px width'] < 0) == len(t):
            ## Negative dispersion
            t['px width'] = np.abs(t['px width'])
            prevStart = deepcopy(t['px start'])
            t['px start'] = t['px end']
            t['px end'] = prevStart
        elif np.sum(t['px width'] > 0) == len(t):
            ## Positive dispersion
            pass
        else:
            raise Exception("Detected zero or flipped dispersion")

        return t
    
    def make_native_px_grid(self,dispPixels=None,
                            doublePx=False,
                            halfpx_correction=True):
        """
        Make a wavelength grid at native resolution

        Parameters
        ----------
        dispPixels: 2 element list or None
            Start and end pixels. If None, it will use the 
            file from the parameters "dispPixels"
        
        doublePx: bool
            Double up pixels?

        halfpx_correction: bool
            Correct for the fact that the wavelengths are mid-px
            so the wavelength start should be the left pixel edge
        """
        if dispPixels is None:
            pxStart = self.param['dispPixels'][0]
            pxEnd = self.param['dispPixels'][1]
        else:
            pxStart = dispPixels[0]
            pxEnd = dispPixels[1]
        
        t = Table()
        if doublePx == True:
            t['px start'] = np.arange(pxStart,pxEnd,2)
            t['px end'] = np.arange(pxStart+2,pxEnd+1,2)
        else:
            t['px start'] = np.arange(pxStart,pxEnd)
            t['px end'] = np.arange(pxStart+1,pxEnd+1)
        if halfpx_correction == True:
            correction = -0.5
        else:
            correction = 0.0
        t['wave start'] = self.wavecal(t['px start'] + correction)
        t['wave end'] = self.wavecal(t['px end'] + correction)
        t['px mid'] = (t['px start'] + t['px end'])/2. + correction
        t['px width'] = t['px end'] - t['px start']
        t['wave mid'] = (t['wave start'] + t['wave end'])/2.
        t['wave width'] = t['wave end'] - t['wave start']

        ## Check whether dispersion is negative
        
        if np.sum(t['px width'] < 0) == len(t):
            ## Negative dispersion
            t['px width'] = np.abs(t['px width'])
            prevStart = deepcopy(t['px start'])
            t['px start'] = t['px end']
            t['px end'] = prevStart
        elif np.sum(t['px width'] > 0) == len(t):
            ## Positive dispersion
            pass
        else:
            raise Exception("Detected zero or flipped dispersion")

        return t

    def make_constant_Rgrid(self,wStart=None,wEnd=None,Rfixed=100,plotBins=True):
        """
        Make an approximately constant R grid rounded to whole pixels

        Parameters
        ----------
        wStart: float
            The wavelength start for bin edges
        wEnd: float
            The wavelength end for bin edges
        Rfixed: float
            The spectral resolution
        plotBins: bool
            Plot the bins over a stellar spectrum?
        """
        if wStart is None:
            wStart = self.wavecal(self.param['dispPixels'][0])
        if wEnd is None:
            wEnd = self.wavecal(self.param['dispPixels'][1])
        wMids, wWidths = make_const_R_grid(wStart=wStart,wEnd=wEnd,Rfixed=Rfixed)
        t = self.find_px_bins_from_waves(wMids,wWidths)
        binEdges = np.append(t['wave start'][0],
                             t['wave end'])
        if plotBins == True:
            avgX, avgY, avgYerr = self.get_avg_spec()
            plt.plot(self.wavecal(avgX),avgY)
            for oneEdge in binEdges:
                plt.axvline(oneEdge)
        return t

    def make_constant_SNRgrid(self,nBins=60,wStart=None,wEnd=None,plotBins=True,
                            iterations=4):
        """
        Make an approximately constant SNR grid rounded to whole pixels

        Parameters
        ----------
        wStart: float
            The wavelength start for bin edges
        wEnd: float
            The wavelength end for bin edges
        nBins: int
            How many bins to make?
        plotBins: bool
            Plot the bins over a stellar spectrum?
        iterations: int
            Number of iterations to use
        """
        st_sp1x,st_sp1y,st_sp1yerr = self.get_avg_spec()
        st_sp1w = self.wavecal(st_sp1x)
        snr_arr = st_sp1y/st_sp1yerr

        wMids, wWidths = make_const_SNR_grid(st_sp1w,snr_arr,nBins,
                                                wStart=wStart,wEnd=wEnd,
                                                plotBins=plotBins,
                                                iterations=iterations)
        
        t = self.find_px_bins_from_waves(wMids,wWidths)

        return t

    def fluxcal(self,recalculate=False):
        """
        Read in the flux cal from file or execute if file doesn't exit


        Parameters
        -----------
        recalculate: bool
            Re-calculate the flux calibration?
        """
        if ((os.path.exists(self.fluxCal_path) == False) |
            (recalculate == True)):
            fluxDat = self.do_fluxcal()
        else:
            fluxDat = ascii.read(self.fluxCal_path)
        return fluxDat


    def do_fluxcal(self):
        """
        Return a flux calibrated spectrum from the average spectrum
        Currently set up for JWST NIRCam using P330-E observations
        """
        modelName = 'p330e_mod_008.fits'
        modelURL = ('https://archive.stsci.edu/hlsps/reference-atlases/cdbs'+
                    '/current_calspec/'+modelName)
        fluxCalModel_path = os.path.join(self.baseDir,
                                         'stellar_models',
                                         modelName)
        if os.path.exists(fluxCalModel_path) == False:
            import urllib.request
            urllib.request.urlretrieve(modelURL,
                                       fluxCalModel_path)
        
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', u.UnitsWarning)
            
            modDat = Table.read(fluxCalModel_path)
        
        xavg, yavg, yerr = self.get_avg_spec()
        wavg = self.wavecal(xavg)
        xavg_edges = xavg - 0.5
        xavg_edges = np.append(xavg_edges,xavg[-1] + 0.5)
        wavg_edges = self.wavecal(xavg_edges)
        ymod_bin = phot_pipeline.do_binning(modDat['WAVELENGTH']/1e4,
                                            modDat['FLUX'],
                                            nBin=wavg_edges)
        if os.path.exists(self.specFile) == False:
            raise Exception("Couldn't find specFile."+
                            "Try running extraction with do_extraction")
        origHead = fits.getheader(self.specFile,extname='ORIG HEADER')
        if 'tshAutoVersion' in self.param:
            autoParam = self.param['tshAutoVersion']
        else:
            autoParam = 1
        
        if (('INSTRUME' not in origHead) | ('FILTER' not in origHead) |
            ('PUPIL' not in origHead)):
            raise NotImplementedError("Couldn't find necessary header keywords")
        elif ((origHead['INSTRUME'] == 'NIRCAM') & 
              (origHead['FILTER'] == 'F322W2') &
              (origHead['PUPIL'] == 'GRISMR')):
            
            specPath_std_rel = ('parameters/spec_params/jwst/prog_06606/'+
                                'spec_nrc_prog06606014001_P330-E_F322W2_' +
                                'autoparam_{:03d}.yaml'.format(autoParam))
        elif ((origHead['INSTRUME'] == 'NIRCAM') & 
              (origHead['FILTER'] == 'F444W') &
              (origHead['PUPIL'] == 'GRISMR')):
            
            specPath_std_rel = ('parameters/spec_params/jwst/prog_06606/'+
                               'spec_nrc_prog06606013001_P330-E_F444W_'+
                               'autoparam_{:03d}.yaml'.format(autoParam))
        else:
            raise NotImplemented("Haven't implemented this instrument config yet")

        specPath_std = os.path.join(self.baseDir,specPath_std_rel)
        if os.path.exists(specPath_std) == False:
            raise Exception("Couldn't find {}".format(specPath_std))
        
        spec_std = spec(specPath_std)

        std_avg = spec_std.get_avg_spec()
        wav_std = spec_std.wavecal(std_avg[0])
        assert np.allclose(wav_std,wavg,atol=1e-5), 'wavelength grid of std is different'

        origHead_std = fits.getheader(spec_std.specFile,extname='ORIG HEADER')
        conv_mult_std = spec_std.countrate_to_electrons_mult(origHead_std)
        std_avg_rate = (std_avg[1] / conv_mult_std)
        calFac =  ymod_bin[1] / std_avg_rate

        yrate = yavg  / self.countrate_to_electrons_mult(origHead)
        ycal = calFac * yrate
        flam_units = u.erg/(u.s * u.cm**2 * u.AA)
        fluxDat = Table()
        fluxDat['wave'] = wavg
        fluxDat['flux'] = ycal * flam_units
        fluxDat['countRate'] = yrate * u.electron / u.s
        fluxDat['cal model'] = ymod_bin[1] * flam_units
        fluxDat['calFac'] = calFac * flam_units / (u.electron / u.s)
        
        fluxDat.meta['outName'] = self.fluxCal_name
        
        phot_pipeline.ensure_directories_are_in_place(self.fluxCal_path)
        fluxDat.write(self.fluxCal_path,overwrite=True)

        return fluxDat
    
    def do_fluxcal2D(self,fluxCal_path=None):
        """
        Flux
        """
        specHDU = fits.open(self.specFile)
        origHead = specHDU['ORIG HEADER'].header
        primHead = specHDU[0].header
        nSrc = primHead['NSOURCE']
        nImg = primHead['NIMG']

        if fluxCal_path is None:
            fluxCal_path = self.fluxCal_path
        fluxCal1D = ascii.read(fluxCal_path)
        rateFactor = u.electron /( self.countrate_to_electrons_mult(origHead) * u.s)
        calFactor = np.tile(fluxCal1D['calFac'],[nSrc,nImg,1]) * rateFactor
        extToMult = ['OPTIMAL SPEC','OPT SPEC ERR','SUM SPEC','SUM SPEC ERR',
                     'BACKGROUND SPEC']
        for oneExtension in extToMult:
            specHDU[oneExtension].data = np.array(specHDU[oneExtension].data * calFactor)
            specHDU[oneExtension].header['BUNIT'] = str(calFactor.unit)
        
        specHDU.writeto(self.fluxCal2D_path,overwrite=True)
        


class batch_spec(phot_pipeline.batchPhot):
    def __init__(self,batchFile='parameters/spec_params/example_batch_spec_parameters.yaml'):
        self.alreadyLists = {'starPositions': 1,'bkgRegionsX': 2, 'bkgRegionsY': 2,
                             'dispPixels': 1, 'excludeList': 1, 'bkgSubDirections': 1}
        self.general_init(batchFile=batchFile)
    
    def make_pipe_obj(self,directParam):
        """
        Make a spectroscopy pipeline object that will be executed in batch
        """
        return spec(directParam=directParam)
    
    def run_all(self,useMultiprocessing=False):
        self.batch_run('showStarChoices')
        self.batch_run('do_extraction',useMultiprocessing=useMultiprocessing)
        self.batch_run('plot_dynamic_spec',saveFits=True)
    
    def plot_all(self):
        self.batch_run('plot_one_spec',showPlot=False)
    
    def test_apertures(self):
        raise NotImplementedError('still working on apertures')
    
    def return_phot_obj(self,ind=0):
        print("try return spec obj")
        
    def return_spec_obj(self,ind=0):
        """
        Return a photometry object so other methods and attributes can be explored
        """
        return spec(directParam=self.paramDicts[ind])
    

def get_spectrum(specFile,specType='Optimal',ind=None,src=0):
    
    HDUList = fits.open(specFile)
    head = HDUList['OPTIMAL Spec'].header
    nImg = head['NIMG']
    
    if ind == None:
        ind = nImg // 2
        
    x = HDUList['DISP INDICES'].data
    y = HDUList['{} SPEC'.format(specType).upper()].data[src,ind,:]
    if specType == 'Optimal':
        fitsExtensionErr = 'OPT SPEC ERR'
    else:
        fitsExtensionErr = 'SUM SPEC ERR'
    
    yerr = HDUList[fitsExtensionErr].data[src,ind,:]
    
    HDUList.close()
    
    return x, y, yerr

if 'TSHIRT_DATA' in os.environ:
    baseDir = os.environ['TSHIRT_DATA']
else:
    baseDir = '.'
comparisonFileNames = glob.glob(os.path.join(baseDir,'tser_data','spec','spec_o9*.fits'))

def compare_spectra(fileNames=comparisonFileNames,specType='Optimal',showPlot=False,
                    normalize=False):
    """
    Compare multiple spectra
    
    Parameters
    ----------
    specType: str
        Which spectrum extension to read? (eg. 'Optimal' vs 'Sum')
    showPlot: bool
        Render the matplotlib plot? If True, it is rendered as a matplotlib
        widget, not saved. If False, the plot is saved to file only
    normalize: bool
        Normalize all the spectra by dividing by the median first?
    
    """
    fig, ax = plt.subplots()
    for oneFile in fileNames:
        x, y, yerr = get_spectrum(oneFile,specType=specType)
        head = fits.getheader(oneFile)
        if normalize == True:
            yShow = y / np.nanmedian(y)
        else:
            yShow = y
        ax.plot(x,yShow,label=head['SRCNAME'])
    ax.legend()
    if showPlot == True:
        plt.show()
    else:
        outPath = os.path.join(baseDir,'plots','spectra','comparison_spec','comparison_spec.pdf')
        fig.savefig(outPath)
        plt.close(fig)
        
def make_const_R_grid(wStart=2.45,wEnd=3.96,Rfixed=100):
    """
    Make a constant-resolution grid
    
    for F444W NIRCam at R=100, I used
    wStart=3.911238, wEnd=5.0,Rfixed=100
    
    Parameters
    ----------
    wStart: float
        The wavelength start for bin edges
    wEnd: float
        The wavelength end for bin edges
    Rfixed: float
        The spectral resolution
    """
    wCurrent = wStart
    binEdges = []
    iterNum=0
    while (iterNum < 1e6) & (wCurrent < wEnd):
        binEdges.append(wCurrent)
        iterNum = iterNum + 1
        wCurrent = wCurrent + wCurrent/Rfixed

    wStarts = np.array(binEdges[0:-1])
    wEnds = np.array(binEdges[1:])
    wMids = (wStarts + wEnds)/2.
    wWidths = wEnds - wStarts

    return wMids, wWidths

def make_const_SNR_grid(x,snr_arr,nBins=60,
                        wStart=None,wEnd=None,
                        plotBins=True,
                        iterations=4,
                        snr_edge_threshold=0.01):
    """
    Make a constant-SNR grid

    Parameters
    ----------
    x: numpy float array
        Wavelengths of spectrum
    snr: numpy float array
        The Signal to noise of the spectrum
    nBins: the number of bins to make
        The spectral resolution
    wStart: float (optional)
        The wavelength start for bin edges
    wEnd: float (optional)
        The wavelength end for bin edges
    iterations: int
        Number of iterations to use
    snr_edge_threshold: float
        How many times the median SNR to use for bin edges
    """
    
    snr_interp = interp1d(x,snr_arr)
    useful_pts = snr_arr > snr_edge_threshold * np.nanmedian(snr_arr)

    if wStart is None:
        wStart = np.nanmin(x[useful_pts])
    if wEnd is None:
        wEnd = np.nanmax(x[useful_pts])
    wEval = np.linspace(wStart,wEnd,nBins)
    for gridIter in np.arange(iterations):
        dx = 1./snr_interp(wEval)
        dx = (wEnd - wStart) * dx/np.sum(dx)
        xEdges = np.append(wStart,wStart + np.cumsum(dx))
        xGrid_mid = (xEdges[1:] + xEdges[:-1])/2.
        xGrid_widths = np.diff(xEdges)
        wEval = xGrid_mid
    
    if plotBins == True:
        plt.plot(x,snr_arr)
        for oneEdge in xEdges:
            plt.axvline(oneEdge,color='orange')
        plt.xlabel("Wavelength")
        plt.ylabel("SNR")
    return xGrid_mid, xGrid_widths


def moving_average(x, w):
    y_out = np.convolve(x, np.ones(w), 'same') / w
    y_out[0:w] = np.nan
    y_out[-w:] = np.nan
    return y_out

def BB(wave,Temp):
    """
    Evaluate a blackbody

    Parameters
    ----------
    wave: numpy array
        Wavelength
    Temp: quantity 
        Temperature (with units)
    """
    exparg = const.h * const.c / (wave * const.k_B * Temp)
    
    return 2. * const.h * const.c**2/wave**5 * 1./(np.exp(exparg) - 1.)

def TB(wave,intens):
    """ 
    Calculate the Brightness temperature from intensity

    Parameters
    -------------
    wave: numpy array
        Wavelength with units of length
    intens: numpy array
        Intensity, must be in itensity units per wavelength
    """
    
    logarg = 1. + 2. * const.h * const.c**2 / (intens * wave**5)
    
    tb = const.h * const.c / (const.k_B * wave * np.log(logarg))
    return tb.si

def TB_err(wave,intens,dI):
    """ 
    Calculate the Brightness temperature error from intensity and
    intensity error

    Parameters
    -------------
    wave: numpy array
        Wavelength with units of length
    intens: numpy array
        Intensity, must be in itensity units per wavelength
    dI: numpy array
        Intensity erorr, must be in itensity units per wavelength
    """
    logarg = 1. + 2. * const.h * const.c**2 / (intens * wave**5)
    fac2 = (2. * const.h * const.c**2)/(intens**2 * wave**5)
    
    tb_err = TB(wave,intens) / (np.log(logarg) * (logarg)) * fac2 * dI
    
    return tb_err.si

def star_to_planet_units(k):
    """
    Convert observed stellar instensity to planet intensity

    Parameters
    ----------
    k: float
        Planet to star radius ratio
    """
    star_to_planet_units = (1./ k**2) #* u.erg / (u.cm**2 * u.s * u.AA)
    return star_to_planet_units

def bin_spec(xorig,yorig,xout,dxout,
                  yerr=None):
    """
    Bin a spectrum with a for loop (slow)

    Parameters
    ----------
    xorig: numpy array
        wavelength of input
    yorig: numpy array
        flux/intensity of input
    xout: numpy array
        bin centers of output
    dxout: numpy array
        bin widths of output
    yerr: None or numpy array

    Outputs
    -------
    ybin: numpy array
        Binned values
    ybin_err: numpy array
        Bin error. If yerr is supplied, weighted avg in the mean
        If no yerr is supplied, it's stdev/sqrt(N)
    """
    if yerr is None:
        pass
    else:
        weights = 1./yerr**2
    binspec_list = []
    binspec_list_err = []
    for ind in np.arange(len(xout)):
        pts = ((xorig > (xout[ind] - dxout[ind]/2.)) & 
              (xorig <= (xout[ind] + dxout[ind]/2.)))
        
        if np.sum(pts) == 0:
            thisBin = np.nan
            thisBinerr = np.nan
        elif yerr is None:
            thisBin = np.mean(yorig[pts])
            thisBinerr = np.std(yorig[pts])/np.sqrt(np.sum(pts))
        else:
            thisBin = np.sum(yorig[pts] * weights[pts])/np.sum(weights[pts])
            thisBinerr = 1./np.sqrt(np.sum(weights[pts]))

        binspec_list.append(thisBin)
        binspec_list_err.append(thisBinerr)

    return np.array(binspec_list), np.array(binspec_list_err)
    # binEdges = xout - dxout
    # binEdges = np.append(binEdges,xout[-1]+dxout[-1])
    # xbin, ybin, ybinerr = phot_pipeline.do_binning(xorig,y=yorig,
    #                                                nBin=binEdges)
    # return xbin,ybin

def TB_from_fp(wave,fp,istar,k):
    """ 
    Calculate the Brightness temperature from Fp/F*

    Parameters
    -------------
    wave: numpy array
        Wavelength with units of length
    fp: numpy array
        Unitless Fp/F*
    istar: numpy array
        Stellar intensity, must be in itensity units per wavelength
    k: float
        Planet to star radius ratio
    """

    # xbin, star_intens_bin = bin_star_intensity(stmodel['Wavelength']/1e4,
    #                                            stmodel['SpecificIntensity'],
    #                                            wave,wave_width)
    Iplanet = istar * star_to_planet_units(k) * fp
    return TB(wave,Iplanet)

def TB_from_fp_err(wave,fp,fp_err,istar,k):
    """ 
    Calculate the Brightness temperature error
    from Fp/F* and Fp/F* err

    Parameters
    -------------
    wave: numpy array
        Wavelength with units of length
    fp: numpy array
        Unitless Fp/F*
    fp_err: numpy array
        Unitless Fp/F* error
    istar: numpy array
        Stellar intensity, must be in itensity units per wavelength
    k: float
        Planet to star radius ratio
    """
    
    Iplanet = istar * star_to_planet_units(k) * fp
    Iplanet_err = istar * star_to_planet_units(k) * fp_err
    
    return TB_err(wave,Iplanet,Iplanet_err)
