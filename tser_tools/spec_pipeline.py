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
import astropy
from astropy.table import Table
from astropy.stats import LombScargle
import multiprocessing
from multiprocessing import Pool
import phot_pipeline


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
    
    def add_parameters_to_header(self,header=None):
        if header is None:
            header = fits.Header()
        
        defaultParams = read_yaml('parameters/spec_params/default_params.yaml')
        
        ## max depth to dig in lists of lists of lists...
        maxDepth = 3
        for oneKey in np.sort(defaultParams.keys()):
            if len(oneKey) > 8:
                keyName = oneKey[0:8]
            else:
                keyName = oneKey
            
            metaDatum = defaultParams[oneKey]
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
        
        if useMultiprocessing == True:
            outputSpec = phot_pipeline.run_multiprocessing_phot(self,fileCountArray,method='spec_for_one_file')
        else:
            outputSpec = []
            for ind in fileCountArray:
                if np.mod(ind,15) == 0:
                    print("Working on {} of {}".format(ind,self.nImg))
                outputSpec.append(self.spec_for_one_file(ind))
        
        timeArr = []
        dispPixelArr = outputSpec[0]['disp indices']
        nDisp = len(dispPixelArr)
        optSpec = np.zeros([self.nImg,nDisp,self.nsrc])
        optSpec_err = np.zeros_like(optSpec)
        sumSpec = np.zeros_like(optSpec)
        sumSpec_err = np.zeros_like(optSpec)
        
        for ind in fileCountArray:
            specDict = outputSpec[ind]
            timeArr.append(specDict['t0'])
            optSpec[ind,:,:] = specDict['opt spec']
            optSpec_err[ind,:,:] = specDict['opt spec err']
            sumSpec[ind,:,:] = specDict['sum spec']
            sumSpec_err[ind,:,:] = specDict['sum spec err']
        
        hdu = fits.PrimaryHDU(optSpec)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with spectroscopy')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('src','source axis')
        hdu.header['AXIS2'] = ('disp','dispersion axis')
        hdu.header['AXIS3'] = ('image','image axis')
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
        
        hduDispIndices = fits.ImageHDU(dispPixelArr)
        hduDispIndices.header['AXIS1'] = ('disp index', 'dispersion index (pixels)')
        hduDispIndices.name = 'Disp Indices'
        
        hduFileNames = self.make_filename_hdu()
        
        ## Get an example original header
        exImg, exHeader = self.get_default_im()
        hduOrigHeader = fits.ImageHDU(None,exHeader)
        hduOrigHeader.name = 'Orig Header'
        
        HDUList = fits.HDUList([hdu,hduOptErr,hduSum,hduSumErr,hduDispIndices,
                                hduFileNames,hduOrigHeader])
        HDUList.writeto(self.specFile,overwrite=True)
        
    
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
    
    def profile_normalize(self,img):
        """
        Renormalize a profile along the spatial direction
        
        Parameters
        -----------
        img: numpy array
            The input profile image to be normalized
        
        """
        normArr = np.sum(img,self.spatialAx)
        
        if self.param['dispDirection'] == 'x':
            norm2D = np.tile(normArr,[img.shape[0],1])
        else:
            norm2D = np.tile(normArr,[img.shape[1],1]).transpose()
        
        norm_profile = np.zeros_like(img)
        nonZero = img != 0
        norm_profile[nonZero] = img[nonZero]/norm2D[nonZero]
        return norm_profile
    
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
            
            ## Renormalize            
            norm_profile = self.profile_normalize(profile_img)
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
            profile_img = profile_img_list[oneSrc]
            smooth_img = smooth_img_list[oneSrc]
            
            ## Find the bad pixels and their missing weights
            badPx = np.abs(smooth_img - img) > 100. * np.sqrt(varImg)
            badPx = badPx | (np.isfinite(img) == False)
            holey_profile = deepcopy(profile_img)
            holey_profile[badPx] = 0.
            holey_weights = np.sum(holey_profile,self.spatialAx)
            correctionFactor = np.ones_like(holey_weights)
            goodPts = holey_weights > 0.
            correctionFactor[goodPts] = 1./holey_weights[goodPts]
            
            if saveFits == True:
                if self.param['dispDirection'] == 'x':
                    correct2D = np.tile(correctionFactor,[img.shape[0],1])
                else:
                    correct2D = np.tile(correctionFactor,[img.shape[1],1]).transpose()
                correct2D[badPx] = 0.
                
                primHDU_prof2Dh = fits.PrimaryHDU(holey_profile)
                holey_profile_name = 'diagnostics/profile_fit/{}_holey_profile_{}.fits'.format(prefixName,oneSrc)
                primHDU_prof2Dh.writeto(holey_profile_name,overwrite=True)
                
                corrHDU = fits.PrimaryHDU(correct2D)
                correction2DName = 'diagnostics/profile_fit/{}_correct_2D_{}.fits'.format(prefixName,oneSrc)
                corrHDU.writeto(correction2DName,overwrite=True)
            
            srcMask = profile_img > 0.
            
            optflux = (np.nansum(imgSub * profile_img * correctionFactor/ varImg,spatialAx) / 
                       np.nansum(profile_img**2/varImg,spatialAx))
            varFlux = (np.nansum(profile_img * correctionFactor,spatialAx) / 
                       np.nansum(profile_img**2/varImg,spatialAx))
            sumFlux = np.nansum(imgSub * srcMask,spatialAx)
            sumErr = np.sqrt(np.nansum(varImg * srcMask,spatialAx))
            
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
    
    def norm_spec(self,x,y,numSplineKnots=None):
        """ Normalize spec """
        dispStart = self.param['dispPixels'][0]
        dispEnd = self.param['dispPixels'][1]
        if numSplineKnots is None:
            numSplineKnots = self.param['numSplineKnots']
        
        knots = np.linspace(dispStart,dispEnd,numSplineKnots)[1:-1]
        spline1 = phot_pipeline.robust_poly(x,np.log10(y),self.param['splineSpecFitOrder'],
                                            knots=knots,useSpline=True)
        modelF = 10**spline1(x)
        return y / modelF
    
    def plot_one_spec(self,src=0,ind=None,specTypes=['Sum','Optimal'],
                      normalize=False,numSplineKnots=None,savePlot=False):
        
        fig, ax = plt.subplots()
        
        for oneSpecType in specTypes:
            x, y, yerr = self.get_spec(specType=oneSpecType,ind=ind,src=src)
            if normalize==True:
                y = self.norm_spec(x,y,numSplineKnots=numSplineKnots)
            ax.plot(x,y,label=oneSpecType)
        ax.legend()
        ax.set_xlabel("{} pixel".format(self.param['dispDirection'].upper()))
        ax.set_ylabel("Counts (e$^-$)")
        if savePlot == True:
            fig.savefig('plots/spectra/individual_spec/{}_ind_spec_{}.pdf'.format(self.param['srcNameShort'],
                                                                                  self.param['nightName']))
        else:
            plt.show()
        plt.close(fig)
        
    def periodogram(self,src=0,ind=None,specType='Optimal',savePlot=False):
        x, y, yerr = self.get_spec(specType=specType,ind=ind,src=src)
        
        normY = self.norm_spec(x,y,numSplineKnots=40)
        #x1, x2 = 
        pts = np.isfinite(normY)
        ls = LombScargle(x[pts],normY[pts],yerr[pts])
        frequency, power = ls.autopower()
        period = 1./frequency
        
        fig, ax = plt.subplots()
        
        ax.loglog(frequency,power)
        ax.set_xlabel('Frequency (1/px)')
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
        
        localPts = frequency < 0.05
        argmax = np.argmax(power[localPts])
        freqAtMax = frequency[localPts][argmax]
        print('Freq at local max power = {}'.format(freqAtMax))
        print('Corresponding period = {}'.format(1./freqAtMax))
        if astropy.__version__ > "3.0":
            print("FAP at local max = {}".format(ls.false_alarm_probability(power[localPts][argmax])))
        else:
            warnings.warn('Not calculating FAP for older versions of astropy')
        
        if savePlot == True:
            periodoName = '{}_spec_periodo_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
            fig.savefig('plots/spectra/periodograms/{}'.format(periodoName))
        else:
            plt.show()
    
    def get_spec(self,specType='Optimal',ind=None,src=0):
        if os.path.exists(self.specFile) == False:
            self.do_extraction()
        
        x, y, yerr = get_spectrum(self.specFile,specType=specType,ind=ind,src=src)
        return x, y, yerr
    
def get_spectrum(specFile,specType='Optimal',ind=None,src=0):
    
    HDUList = fits.open(specFile)
    head = HDUList['OPTIMAL Spec'].header
    nImg = head['NIMG']
    
    if ind == None:
        ind = nImg // 2
        
    x = HDUList['DISP INDICES'].data
    y = HDUList['{} SPEC'.format(specType).upper()].data[ind,:,src]
    if specType == 'Optimal':
        fitsExtensionErr = 'OPT SPEC ERR'
    else:
        fitsExtensionErr = 'SUM SPEC ERR'
    
    yerr = HDUList[fitsExtensionErr].data[ind,:,src]
    
    HDUList.close()
    
    return x, y, yerr
