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
        self.check_parameters()
        
    def check_parameters(self):
        pass
    
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
            origName = 'diagnostics//spec_backsub/{}_for_backsub_{}.fits'.format(prefixName,oneDirection)
            primHDU.writeto(origName,overwrite=True)
            primHDU_mod = fits.PrimaryHDU(bkgModel)
            subModelName = 'diagnostics/spec_backsub/{}_backsub_model_{}.fits'.format(prefixName,oneDirection)
            primHDU_mod.writeto(subModelName,overwrite=True)
            subName = 'diagnostics/spec_backsub/{}_subtracted_{}.fits'.format(prefixName,oneDirection)
            subHDU = fits.PrimaryHDU(img - bkgModel,outHead)
            subHDU.writeto(subName,overwrite=True)
        
        return img - bkgModel, outHead
    
    def do_backsub(self,img,head,ind=None,saveFits=False):
        subImg = img
        subHead = head
        for oneDirection in ['Y','X']:
            if self.param['bkgSub{}'.format(oneDirection)] == True:
                subImg, subHead = self.backsub_oneDir(subImg,subHead,oneDirection,ind=ind,saveFits=saveFits)
        return subImg
    
    def spec_for_one_file(ind):
        """ Get spectroscopy for one file """
        
        oneImg = self.fileL[ind]
        img, head = self.getImg(oneImgName)
        t0 = get_date(head)
        readNoise = get_read_noise(self,head)
        err = np.sqrt(np.abs(img) + readNoise**2) ## Should already be gain-corrected
        
        imgSub = self.do_backsub(img,head,ind)
        