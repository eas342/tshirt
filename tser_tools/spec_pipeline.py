import photutils
from astropy.io import fits, ascii
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
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
from astropy.table import Table
import multiprocessing
from multiprocessing import Pool
import phot_pipeline


maxCPUs = multiprocessing.cpu_count() // 3

def read_yaml(filePath):
    with open(filePath) as yamlFile:
        yamlStructure = yaml.safe_load(yamlFile)
    return yamlStructure

class spec(phot_pipeline.phot):
    def __init__(self,paramFile='parameters/spec_params/example_spec_parameters.yaml'):
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
        """
        
        self.param = read_yaml(paramFile)
        defaultParams = read_yaml('parameters/spec_params/default_params.yaml')
        
        for oneKey in defaultParams.keys():
            if oneKey not in self.param:
                self.param[oneKey] = defaultParams[oneKey]
        
        # Get the file list
        self.fileL = self.get_fileList()
        self.nImg = len(self.fileL)
        
        self.nsrc = len(self.param['starPositions'])
        
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
        self.srcNames[0] = 'src'
        
        ## Set up file names for output
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        self.specFile = 'tser_data/spec/spec_'+self.dataFileDescrip+'.fits'
        #self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'
        #self.refCorPhotFile = 'tser_data/refcor_phot/refcor_'+self.dataFileDescrip+'.fits'
        self.get_summation_direction()
        
        self.check_parameters()
        
    def check_parameters(self):
        dispCheck = (self.param['dispDirection'] == 'x') | (self.param['dispDirection'] == 'y')
        assert dispCheck, 'Dispersion direction parameter not valid'
        
    
    def get_summation_direction(self):
        if self.param['dispDirection'] == 'x':
            self.spatialAx = 0 ## summation axis along Y (spatial axis)
            self.dispAx = 1 ## dispersion axis is X
        else:
            self.spatialAx = 1 ## summation axis along X (spatial axis)
            self.dispAx = 0 ## dispersion axis is 0
    
    def do_extraction(self):
        """
        Extract all spectroscopy
        """
        fileCountArray = np.arange(len(self.fileL))
        for ind in fileCountArray:
            outputSpec.append(self.spec_for_one_file(ind))
    
    def backsub_oneDir(self,img,head,oneDirection,saveFits=False,
                       showEach=False,ind=None):
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
        """
        
        if oneDirection == 'X':
            subtractionIndexArray = np.arange(img.shape[1])
            cross_subtractionIndexArray = np.arange(img.shape[0])
        elif oneDirection == 'Y':
            subtractionIndexArray = np.arange(img.shape[0])
            cross_subtractionIndexArray = np.arange(img.shape[1])
        else:
            raise Exception("Unrecognized subtraction direction")
        
        ## set up which points to do background fitting for
        pts = np.zeros(len(subtractionIndexArray),dtype=np.bool)
        for oneRegion in self.param['bkgRegions{}'.format(oneDirection)]:
            pts[oneRegion[0]:oneRegion[1]] = True
        
        fitOrder = self.param['bkgOrder{}'.format(oneDirection)]
        ## make a background model
        bkgModel = np.zeros_like(img)
        for cross_Ind in cross_subtractionIndexArray:
            ind_var = subtractionIndexArray ## independent variable
            if oneDirection == 'X':
                dep_var = img[cross_Ind,:]
            else:
                dep_var = img[:,cross_Ind]
            polyFit = phot_pipeline.robust_poly(ind_var[pts],dep_var[pts],fitOrder)
            dep_var_model = np.polyval(polyFit,ind_var)
            
            if oneDirection == 'X':
                bkgModel[cross_Ind,:] = dep_var_model
            else:
                bkgModel[:,cross_Ind] = dep_var_model
                
            if showEach == True:
                plt.plot(ind_var,dep_var,label='data')
                plt.plot(ind_var[pts],dep_var[pts],'o',color='red',label='pts fit')
                plt.plot(ind_var,dep_var_model,label='model')
                plt.show()
        
        outHead = deepcopy(head)
        if oneDirection == 'X':
            outHead['ROWSUB'] = (True, "Is row-by-row subtraction performed?")
        else:
            outHead['COLSUB'] = (True, "Is row-by-row subtraction performed?")
        
        if saveFits == True:
            primHDU = fits.PrimaryHDU(img,head)
            if ind == None:
                prefixName = 'unnamed'
            else:
                prefixName = os.path.splitext(os.path.basename(self.fileL[ind]))[0]
            origName = 'diagnostics/spec_backsub/{}_for_backsub_{}.fits'.format(prefixName,oneDirection)
            primHDU.writeto(origName,overwrite=True)
            primHDU_mod = fits.PrimaryHDU(bkgModel)
            subModelName = 'diagnostics/spec_backsub/{}_backsub_model_{}.fits'.format(prefixName,oneDirection)
            primHDU_mod.writeto(subModelName,overwrite=True)
            subName = 'diagnostics/spec_backsub/{}_subtracted_{}.fits'.format(prefixName,oneDirection)
            subHDU = fits.PrimaryHDU(img - bkgModel,outHead)
            subHDU.writeto(subName,overwrite=True)
        
        return img - bkgModel, bkgModel, outHead
    
    def do_backsub(self,img,head,ind=None,saveFits=False):
        subImg = img
        subHead = head
        bkgModelTotal = np.zeros_like(subImg)
        for oneDirection in ['Y','X']:
            if self.param['bkgSub{}'.format(oneDirection)] == True:
                subImg, bkgModel, subHead = self.backsub_oneDir(subImg,subHead,oneDirection,
                                                                ind=ind,saveFits=saveFits)
                bkgModelTotal = bkgModelTotal + bkgModel
        return subImg, bkgModelTotal, subHead
    
    def find_profile(self,img,head,ind=None,saveFits=False,showEach=False):
        """
        Find the spectroscopic profile using splines along the spectrum
        This assumes an inherently smooth continuum (like a stellar source)
        """
        
        dispStart = self.param['dispPixels'][0]
        dispEnd = self.param['dispPixels'][1]
        ind_var = np.arange(dispStart,dispEnd) ## independent variable
        knots = np.linspace(dispStart,dispEnd,self.param['numSplineKnots'])[1:-1]
        
        profile_img_list = []
        smooth_img_list = [] ## save the smooth version if running diagnostics
        
        for oneSourcePos in self.param['starPositions']:
            profile_img = np.zeros_like(img)
            startSpatial = int(oneSourcePos - self.param['apWidth'] / 2.)
            endSpatial = int(oneSourcePos + self.param['apWidth'] / 2.)
            for oneSpatialInd in np.arange(startSpatial,endSpatial + 1):
                if self.param['dispDirection'] == 'x':
                    dep_var = img[oneSpatialInd,dispStart:dispEnd]
                else:
                    dep_var = img[dispStart:dispEnd,oneSpatialInd]
                
                spline1 = phot_pipeline.robust_poly(ind_var,np.log10(dep_var),self.param['splineSpecFitOrder'],
                                                    knots=knots,useSpline=True)
                modelF = 10**spline1(ind_var)
                
                if showEach == True:
                    plt.plot(ind_var,dep_var,'o',label='data')
                    plt.plot(ind_var,modelF,label='model')
                    plt.show()
                    pdb.set_trace()
                
                if self.param['dispDirection'] == 'x':
                    profile_img[oneSpatialInd,dispStart:dispEnd] = modelF
                else:
                    profile_img[dispStart:dispEnd,oneSpatialInd] = modelF
            
            ## Find obviously bad pixels
            smooth_img = deepcopy(profile_img)
            
            ## Experimenting with weight bad px as 0
            # badPx = np.abs(profile_img - img) > 100. * np.sqrt(profile_img)
            # profile_img[badPx] = 0.
            
            ## Renormalize
            normArr = np.sum(profile_img,self.spatialAx)
            
            if self.param['dispDirection'] == 'x':
                norm2D = np.tile(normArr,[img.shape[0],1])
            else:
                norm2D = np.tile(normArr,[img.shape[1],1]).transpose()
            
            norm_profile = np.zeros_like(profile_img)
            nonZero = profile_img != 0
            norm_profile[nonZero] = profile_img[nonZero]/norm2D[nonZero]
            
            profile_img_list.append(norm_profile)
            ## save the smoothed image
            smooth_img_list.append(smooth_img)
                
        if saveFits == True:
            primHDU = fits.PrimaryHDU(img,head)
            if ind == None:
                prefixName = 'unnamed'
            else:
                prefixName = os.path.splitext(os.path.basename(self.fileL[ind]))[0]
            origName = 'diagnostics/profile_fit/{}_for_profile_fit.fits'.format(prefixName)
            primHDU.writeto(origName,overwrite=True)
            for ind,profile_img in enumerate(profile_img_list):
                ## Saved the smoothed model
                primHDU_smooth = fits.PrimaryHDU(smooth_img_list[ind])
                smoothModelName = 'diagnostics/profile_fit/{}_smoothed_src_{}.fits'.format(prefixName,ind)
                primHDU_smooth.writeto(smoothModelName,overwrite=True)
                
                ## Save the profile
                primHDU_mod = fits.PrimaryHDU(profile_img)
                profModelName = 'diagnostics/profile_fit/{}_profile_model_src_{}.fits'.format(prefixName,ind)
                primHDU_mod.writeto(profModelName,overwrite=True)
        
        
        return profile_img_list, smooth_img_list
    
    def spec_for_one_file(self,ind,saveFits=False):
        """ Get spectroscopy for one file """
        
        oneImgName = self.fileL[ind]
        img, head = self.getImg(oneImgName)
        t0 = self.get_date(head)
        
        imgSub, bkgModel, subHead = self.do_backsub(img,head,ind)
        readNoise = self.get_read_noise(head)
        ## Background and read noise only.
        ## Smoothed source flux added below
        varImg = readNoise**2 + bkgModel ## in electrons because it should be gain-corrected
        
        profile_img_list, smooth_img_list = self.find_profile(imgSub,subHead,ind)
        for oneSrc in np.arange(self.nsrc): ## use the smoothed flux for the variance estimate
            varImg = varImg + smooth_img_list[oneSrc]
        
        if saveFits == True:
            prefixName = os.path.splitext(os.path.basename(oneImgName))[0]
            varName = 'diagnostics/variance_img/{}_variance.fits'.format(prefixName)
            primHDU = fits.PrimaryHDU(varImg)
            primHDU.writeto(varName,overwrite=True)
        
        spatialAx = self.spatialAx
        dispAx = self.dispAx
                
        ## dispersion indices in pixels (before wavelength calibration)
        nDisp = img.shape[dispAx]
        dispIndices = np.arange(nDisp)
        
        optSpectra = np.zeros([nDisp,self.nsrc])
        optSpectra_err = np.zeros_like(optSpectra)
        sumSpectra = np.zeros_like(optSpectra)
        sumSpectra_err = np.zeros_like(optSpectra)
        
        for oneSrc in np.arange(self.nsrc):
            ## Insert an interpolation method here
            # badPx = np.abs(profile_img - img) > 100. * np.sqrt(profile_img)
            # profile_img[badPx] = 0.
            
            profile_img = profile_img_list[oneSrc]
            srcMask = profile_img > 0.
            
            optflux = (np.sum(imgSub * profile_img / varImg,spatialAx) / 
                       np.sum(profile_img**2/varImg,spatialAx))
            varFlux = (np.sum(profile_img,spatialAx) / 
                       np.sum(profile_img**2 * varImg,spatialAx))
            sumFlux = np.sum(imgSub * srcMask,spatialAx)
            sumErr = np.sqrt(np.sum(varImg * srcMask,spatialAx))
            
            optSpectra[:,oneSrc] = optflux
            optSpectra_err[:,oneSrc] = np.sqrt(varFlux)
            
            sumSpectra[:,oneSrc] = sumFlux
            sumSpectra_err[:,oneSrc] = sumErr
        
        extractDict = {} ## spectral extraction dictionary
        extractDict['t0'] = t0
        extractDict['disp indices'] = dispIndices
        extractDict['opt spec'] = optSpectra
        extractDict['opt spec err'] = optSpectra_err
        extractDict['sum spec'] = sumSpectra
        extractDict['sum spec err'] = sumSpectra_err
        
        return extractDict
        
        