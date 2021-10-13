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
import pdb
from copy import deepcopy
import yaml
import warnings
from scipy.stats import binned_statistic
import astropy
from astropy.table import Table
from astropy.stats import LombScargle
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
    
    def do_extraction(self,useMultiprocessing=False):
        """
        Extract all spectroscopy
        """
        fileCountArray = np.arange(self.nImg)
        
        if self.param['fixedProfile'] == True:
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
                
        
        hdu = fits.PrimaryHDU(optSpec)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with spectroscopy')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('disp','dispersion axis')
        hdu.header['AXIS2'] = ('image','image axis')
        hdu.header['AXIS3'] = ('src','source axis')
        hdu.header['SRCNAME'] = (self.param['srcName'], 'Source name')
        hdu.header['NIGHT'] = (self.param['nightName'], 'Night Name')
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
         
        HDUList = fits.HDUList([hdu,hduOptErr,hduSum,hduSumErr,
                                hduBack,hduDispIndices,
                                hduTime,hduFileNames,hduOrigHeader,
                                hduRef])
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
            pts = np.zeros(len(subtractionIndexArray),dtype=bool)
            for oneRegion in self.param['bkgRegions{}'.format(oneDirection)]:
                if self.param['mosBacksub'] == True:
                    startSub = int(self.param['starPositions'][srcInd] + oneRegion[0])
                    endSub = int(self.param['starPositions'][srcInd] + oneRegion[1])
                else:
                    startSub, endSub = int(oneRegion[0]),int(oneRegion[1])
                
                pts[startSub:endSub] = True
            
            
            for cross_Ind in cross_subtractionIndexArray:
                ind_var = subtractionIndexArray ## independent variable
                if oneDirection == 'X':
                    dep_var = img[cross_Ind,:]
                else:
                    dep_var = img[:,cross_Ind]
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
            
            ## add the delta in again for profile normalization > 0
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
        oneSourcePos = self.param['starPositions'][src]
        startSpatial = int(oneSourcePos - self.param['apWidth'] / 2.)
        endSpatial = int(oneSourcePos + self.param['apWidth'] / 2.)
        nSpatial = (endSpatial - startSpatial) + 1
        
        ## This will be slow at first because I'm starting with a for loop.
        ## Eventually do fancy 3D matrices to make it fast
        #optflux = np.zeros(nDisp) * np.nan
        #varFlux = np.zeros_like(optflux) * np.nan
        varPure = varImg - readNoise**2 ## remove the read noise because we'll put it in covariance matrix
        weight2D = np.zeros_like(profile_img)
        
        for oneInd in np.arange(nDisp):
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
        
        
        for oneSrc in np.arange(self.nsrc):
            profile_img = profile_img_list[oneSrc]
            smooth_img = smooth_img_list[oneSrc]
            
            ## Find the bad pixels and their missing weights
            finitep = (np.isfinite(img) & np.isfinite(varImg) & np.isfinite(smooth_img))
            badPx = finitep == False ## start by marking NaNs as bad pixels
            ## also mark large deviations from profile fit
            badPx[finitep] = np.abs(smooth_img[finitep] - img[finitep]) > self.param['sigForBadPx'] * np.sqrt(varImg[finitep])
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
            
        
        extractDict = {} ## spectral extraction dictionary
        extractDict['t0'] = t0
        extractDict['disp indices'] = dispIndices
        extractDict['opt spec'] = optSpectra
        extractDict['opt spec err'] = optSpectra_err
        extractDict['sum spec'] = sumSpectra
        extractDict['sum spec err'] = sumSpectra_err
        extractDict['back spec'] = backSpectra
        extractDict['airmass'] = airmass
        
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
            ax.set_xlabel("Wavelength ($\mu$m)")
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
            x_unit = 'Frequency (1/$\lambda^2$)'
        elif transform == 'inv-lam':
            lam = self.wavecal(x_px)
            x = 1./lam
            x_unit = '$\lambda^2/(\Delta \lambda) (\mu$m)'
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
    
    def dyn_specFile(self,src=0):
        return "{}_src_{}.fits".format(self.dyn_specFile_prefix,src)
        
    def plot_dynamic_spec(self,src=0,saveFits=True,specAtTop=True,align=False,
                          alignDiagnostics=False,extraFF=False,
                          specType='Optimal',showPlot=False,
                          vmin=None,vmax=None):
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
            

                
            
            ## Expect a card-too-long warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message= "Card is too long, comment will be truncated")
                outHDUList = fits.HDUList([dynHDU,dynHDUerr,dispHDU,timeHDU,offsetHDU,avgHDU, avgErrHDU])
                if align == True:
                    if specType == 'Optimal':
                        alignedHDU = fits.ImageHDU(useSpec,HDUList['OPTIMAL SPEC'].header)
                    else:
                        alignedHDU = fits.ImageHDU(useSpec,HDUList['SUM SPEC'].header)
                    ## Save the aligned spectra
                    outHDUList.append(alignedHDU)
                outHDUList.writeto(self.dyn_specFile(src),overwrite=True)
        
        if specAtTop == True:
            fig, axArr = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': [1, 3]})
            axTop = axArr[0]
            axTop.plot(waveIndices,avgSpec)
            ax = axArr[1]
        else:
            fig, ax = plt.subplots()
        
        if vmin is None:
            vmin=0.95
        if vmax is None:
            vmax=1.05
        
        imShowData = ax.imshow(dynamicSpec,vmin=vmin,vmax=vmax)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        ax.set_xlabel('Disp (pixels)')
        ax.set_ylabel('Time (Image #)')
        dispPix = np.array(self.param['dispPixels']) + self.dispOffsets[src]
        ax.set_xlim(dispPix[0],dispPix[1])
        fig.colorbar(imShowData,label='Normalized Flux')
        if specAtTop == True:
            ## Fix the axes to be the same
            pos = ax.get_position()
            pos2 = axTop.get_position()
            axTop.set_position([pos.x0,pos2.y0,pos.width,pos2.height])
            
        
        dyn_spec_name = '{}_dyn_spec_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
        dyn_spec_path = os.path.join(self.baseDir,'plots','spectra','dynamic_spectra',dyn_spec_name)
        fig.savefig(dyn_spec_path,bbox_inches='tight')
        
        if showPlot == True:
            fig.show()
        else:
            plt.close(fig)
        
        HDUList.close()
    
    def plot_noise_spectrum(self,src=0,showPlot=True,yLim=None):
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
        """
        fig, ax = plt.subplots()
        HDUList = fits.open(self.dyn_specFile(src))
        x = HDUList['DISP INDICES'].data
        y = np.nanstd(HDUList['DYNAMIC SPEC'].data,axis=0)
        theo_y = np.nanmedian(HDUList['DYN SPEC ERR'].data,axis=0)
        ax.plot(x,y * 100.,label='Measured noise')
        ax.plot(x,theo_y * 100.,label='Theoretical noise')
        ax.set_xlabel("Disp Pixel")
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
            fig.savefig(outPath,overwrite=True)
    
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
                                       dtype=np.int)
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
                            recalculate=False,align=False,refCorrect=False):
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
            dispSt, dispEnd = np.array(np.array(self.param['dispPixels']) + self.dispOffsets[src],dtype=np.int)
        else:
            dispSt, dispEnd = dispIndices
        
        binEdges = np.array(np.linspace(dispSt,dispEnd,nbins+1),dtype=np.int)
        binStarts = binEdges[0:-1]
        binEnds = binEdges[1:]
        binIndices = np.arange(len(binStarts))
        
        binGrid = np.zeros([nTime,nbins])
        binGrid_err = np.zeros_like(binGrid)
        
        binned_disp = np.zeros(nbins)
        
        #db.set_trace()
        
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
        outHDUList = fits.HDUList([outHDU,errHDU,timeHDU,offsetHDU,dispHDU])
        outHDUList.writeto(self.wavebin_specFile(nbins,src),overwrite=True)
        
        HDUList.close()
    
    def get_offset_time(self,time):
        return np.floor(np.min(time))
    
    def plot_wavebin_series(self,nbins=10,offset=0.005,showPlot=False,yLim=None,xLim=None,
                            recalculate=False,dispIndices=None,differential=False,
                            interactive=False,unit='fraction',align=False,specType='Optimal',
                            src=0,refCorrect=False):
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
        """
        if (os.path.exists(self.wavebin_specFile(nbins=nbins,srcInd=src)) == False) | (recalculate == True):
            self.make_wavebin_series(nbins=nbins,dispIndices=dispIndices,recalculate=recalculate,
                                     specType=specType,align=align,src=src,refCorrect=refCorrect)
        
        HDUList = fits.open(self.wavebin_specFile(nbins=nbins,srcInd=src))
        time = HDUList['TIME'].data
        offset_time = self.get_offset_time(time)
        
        disp = HDUList['DISP INDICES'].data
        
        binGrid = HDUList['BINNED F'].data
        binGrid_err = HDUList['BINNED ERR'].data
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
                p.circle(x='t',y="y{:02d}".format(ind),source=source,color=next(colors))
            
            p.add_tools(HoverTool(tooltips=[('name', '@name'),('index','@ind')]))
            bokeh.plotting.show(p)
            
        else:
            fig, ax = plt.subplots()
            for ind,oneDisp in enumerate(disp):
                if unit == 'fraction':
                    y_plot = binGrid[:,ind] - offset * ind
                else:
                    y_plot = (binGrid[:,ind] - 1.0) * 1e6 - offset * ind * 1e6
                
                ax.errorbar(time - offset_time,y_plot,fmt='o')
            
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
                fig.savefig(outPath)
                plt.close(fig)

            
            HDUList.close()
    
    
    def get_wavebin_series(self,nbins=10,recalculate=False,specType='Optimal',srcInd=0):
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
            self.plot_wavebin_series(nbins=nbins,recalculate=recalculate,specType=specType)
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
                            specType='Optimal',npoints=15,srcInd=0):
        """ 
        Get a table of noise measurements for all wavelength bins
        
        Parameters
        ----------
        nbins: int
            The number of wavelength bins
        shorten: bool
            Use a short segment of the full time series?
            This could be useful for avoiding bad data or a deep transit
        npoints: int
            Number of points to include in the calculation (if shorten is True)
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
        t['Wave (st)'] = np.round(self.wavecal(disp['Bin Start']),3)
        t['Wave (mid)'] = np.round(self.wavecal(t['Disp Mid']),3)
        t['Wave (end)'] = np.round(self.wavecal(disp['Bin End']),3)
        if shorten == True:
            binGrid = binGrid[0:npoints,:]
        t['Stdev (%)'] = np.round(np.std(binGrid,axis=0) * 100.,4)
        t['Theo Err (%)'] = np.round(np.median(binGrid_err,axis=0) * 100.,4)
        
        medianV = np.median(binGrid,axis=0)
        absDeviation = np.abs(binGrid - medianV)
        t['MAD (%)'] = np.round(np.median(absDeviation,axis=0) * 100.,4)
        
        HDUList.close()
        return t
    
    def get_broadband_series(self,src=0):
        HDUList = fits.open(self.specFile)
        t = Table()
        t['time'] = HDUList['TIME'].data
        
        spec2D = HDUList['OPTIMAL SPEC'].data[src]
        spec2D_err = HDUList['OPT SPEC ERR'].data[src]
        t['Flux'] = np.nansum(spec2D,1)
        t['Flux Err'] = np.sqrt(np.nansum(spec2D_err**2,1))
        
        norm_value = np.nanmedian(t['Flux'])
        t['Norm Flux'] = t['Flux']  / norm_value
        t['Norm Flux Err'] = t['Flux Err'] / norm_value
        
        HDUList.close()
        
        return t
    
    def plot_broadband_series(self,src=0,showPlot=True,matchIraf=False):
        t = self.get_broadband_series(src=src)
        offset_time = self.get_offset_time(t['time'])
        
        err_ppm = np.nanmedian(t['Norm Flux Err']) * 1e6
        print('Formal Err = {} ppm '.format(err_ppm))
        fig, ax = plt.subplots()
        
        if matchIraf == True:
            marker = '.'
            x = np.arange(len(t))
            ax.set_xlabel("Int Number")
        else:
            marker = 'o'
            x = t['time'] - offset_time
            ax.set_xlabel("Time - {} (days)".format(offset_time))
            
        
        ax.plot(x,t['Norm Flux'],marker)
        ax.set_ylabel("Normalized Flux")
        if showPlot == True:
            fig.show()
        else:
            bb_series_name = '{}_bb_series_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
            outName = os.path.join(self.baseDir,'plots','spectra','broadband_series',bb_series_name)
            fig.savefig(outName)

    
    
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
            ax.text(xPos+txtOffset,yPos+txtOffset,name,color='white')
        
        if showBack == True:
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
        
    
    def wavecal(self,dispIndices,waveCalMethod=None,head=None,**kwargs):
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
        """
        if waveCalMethod == None:
            waveCalMethod = self.param['waveCalMethod']
        
        if waveCalMethod == None:
            wavelengths = dispIndices
        elif waveCalMethod == 'NIRCamTS':
            if head == None:
                head = fits.getheader(self.specFile,extname='ORIG HEADER')
            wavelengths = instrument_specific.jwst_inst_funcs.ts_wavecal(dispIndices,obsFilter=head['FILTER'],
                                                                         **kwargs)
        elif waveCalMethod == 'simGRISMC':
            wavelengths = instrument_specific.jwst_inst_funcs.ts_grismc_sim(dispIndices,**kwargs)
        
        elif waveCalMethod == 'wfc3Dispersion':
            wavelengths = instrument_specific.hst_inst_funcs.hstwfc3_wavecal(dispIndices,**kwargs)
            
        else:
            raise Exception("Unrecognized wavelength calibration method {}".format(waveCalMethod))
            
        wavelengths = wavelengths - self.param['waveCalOffset']
        return wavelengths
        
        

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
        
