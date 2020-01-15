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
import analysis



class spec(phot_pipeline.phot):
    def __init__(self,paramFile='parameters/spec_params/example_spec_parameters.yaml',
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
        
        defaultParams = phot_pipeline.read_yaml('parameters/spec_params/default_params.yaml')
        
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
        
        self.dyn_specFile_prefix = 'tser_data/dynamic_spec/dyn_spec_{}'.format(self.dataFileDescrip)
        
        self.wavebin_file_prefix = 'tser_data/wavebin_spec/wavebin_spec_{}'.format(self.dataFileDescrip)
        #self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'
        #self.refCorPhotFile = 'tser_data/refcor_phot/refcor_'+self.dataFileDescrip+'.fits'
        self.get_summation_direction()
        
        ## a little delta to add to add to profile so that you don't get log(negative)
        self.floor_delta = self.param['readNoise'] * 2. 
        
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
        
        if useMultiprocessing == True:
            outputSpec = phot_pipeline.run_multiprocessing_phot(self,fileCountArray,method='spec_for_one_file')
        else:
            outputSpec = []
            for ind in fileCountArray:
                outputSpec.append(self.spec_for_one_file(ind))
        
        timeArr = []
        dispPixelArr = outputSpec[0]['disp indices']
        nDisp = len(dispPixelArr)
        optSpec = np.zeros([self.nsrc,self.nImg,nDisp])
        optSpec_err = np.zeros_like(optSpec)
        sumSpec = np.zeros_like(optSpec)
        sumSpec_err = np.zeros_like(optSpec)
        backSpec = np.zeros_like(optSpec)
        
        for ind in fileCountArray:
            specDict = outputSpec[ind]
            timeArr.append(specDict['t0'].jd)
            optSpec[:,ind,:] = specDict['opt spec']
            optSpec_err[:,ind,:] = specDict['opt spec err']
            sumSpec[:,ind,:] = specDict['sum spec']
            sumSpec_err[:,ind,:] = specDict['sum spec err']
            backSpec[:,ind,:] = specDict['back spec']
        
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
        
        hduFileNames = self.make_filename_hdu()
        
        ## Get an example original header
        exImg, exHeader = self.get_default_im()
        hduOrigHeader = fits.ImageHDU(None,exHeader)
        hduOrigHeader.name = 'Orig Header'
        
        ## Save the times
        hduTime = fits.ImageHDU(np.array(timeArr))
        hduTime.header['AXIS1'] = ('time', 'time in Julian Day (JD)')
        hduTime.name = 'TIME'
        
        HDUList = fits.HDUList([hdu,hduOptErr,hduSum,hduSumErr,
                                hduBack,hduDispIndices,
                                hduTime,hduFileNames,hduOrigHeader])
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
                
                ## this is a very long way to calculate a log that avoids runtime warnings
                fitY = np.zeros_like(dep_var) * np.nan
                positivep = np.zeros_like(dep_var,dtype=np.bool)
                finitep = np.isfinite(dep_var)
                positivep[finitep] = (dep_var[finitep] > 0. - self.floor_delta)
                fitY[positivep] = np.log10(dep_var[positivep] + self.floor_delta)
                
                try:
                    spline1 = phot_pipeline.robust_poly(ind_var,fitY,self.param['splineSpecFitOrder'],
                                                        knots=knots,useSpline=True,sigreject=self.param['splineSigRej'],
                                                        plotEachStep=False,preScreen=self.param['splinePreScreen'])
                except ValueError as inst:
                    if str(inst) == 'Interior knots t must satisfy Schoenberg-Whitney conditions':
                        print(inst)
                        print("Trying my debugging mode")
                        
                        print("Value error")
                        pdb.set_trace()
                    else:
                        raise inst
                
                
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
                profile_img[oneSpatialInd,dispStart:dispEnd] += self.floor_delta
            else:
                profile_img[dispStart:dispEnd,oneSpatialInd] += self.floor_delta
            
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
        if np.mod(ind,15) == 0:
            print("On {} of {}".format(ind,len(self.fileL)))
        
        oneImgName = self.fileL[ind]
        img, head = self.getImg(oneImgName)
        t0 = self.get_date(head)
        
        imgSub, bkgModel, subHead = self.do_backsub(img,head,ind)
        readNoise = self.get_read_noise(head)
        ## Background and read noise only.
        ## Smoothed source flux added below
        varImg = readNoise**2 + bkgModel ## in electrons because it should be gain-corrected
        
        profile_img_list, smooth_img_list = self.find_profile(imgSub,subHead,ind,saveFits=saveFits)
        for oneSrc in np.arange(self.nsrc): ## use the smoothed flux for the variance estimate
            varImg = varImg + np.abs(smooth_img_list[oneSrc]) ## negative flux should be approximated as photon noise
        
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
        
        optSpectra = np.zeros([self.nsrc,nDisp])
        optSpectra_err = np.zeros_like(optSpectra)
        sumSpectra = np.zeros_like(optSpectra)
        sumSpectra_err = np.zeros_like(optSpectra)
        backSpectra = np.zeros_like(optSpectra)
        
        for oneSrc in np.arange(self.nsrc):
            profile_img = profile_img_list[oneSrc]
            smooth_img = smooth_img_list[oneSrc]
            
            ## Find the bad pixels and their missing weights
            finitep = np.isfinite(img)
            badPx = finitep == False ## start by marking NaNs as bad pixels
            ## also mark large deviations from profile fit
            badPx[finitep] = np.abs(smooth_img[finitep] - img[finitep]) > self.param['sigForBadPx'] * np.sqrt(varImg[finitep])
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
                markBad = badPx & (profile_img > 0.)
                correct2D[markBad] = 0.
                
                primHDU_prof2Dh = fits.PrimaryHDU(holey_profile)
                holey_profile_name = 'diagnostics/profile_fit/{}_holey_profile_{}.fits'.format(prefixName,oneSrc)
                primHDU_prof2Dh.writeto(holey_profile_name,overwrite=True)
                
                corrHDU = fits.PrimaryHDU(correct2D)
                correction2DName = 'diagnostics/profile_fit/{}_correct_2D_{}.fits'.format(prefixName,oneSrc)
                corrHDU.writeto(correction2DName,overwrite=True)
            
            srcMask = profile_img > 0.
            
            ## Replaced the old lines to avoid runtime warnings
            # optflux = (np.nansum(imgSub * profile_img * correctionFactor/ varImg,spatialAx) /
            #            np.nansum(profile_img**2/varImg,spatialAx))
            # varFlux = (np.nansum(profile_img * correctionFactor,spatialAx) /
            #            np.nansum(profile_img**2/varImg,spatialAx))
            optNumerator = np.nansum(imgSub * profile_img * correctionFactor/ varImg,spatialAx)
            denom =  np.nansum(profile_img**2/varImg,spatialAx)
            nonz = (denom != 0.) & np.isfinite(denom)
            optflux = np.zeros_like(optNumerator) * np.nan
            optflux[nonz] = optNumerator[nonz] / denom[nonz]
            
            varNumerator = np.nansum(profile_img * correctionFactor,spatialAx)
            varFlux = np.zeros_like(varNumerator) * np.nan
            varFlux[nonz] = varNumerator[nonz] / denom[nonz]
            
            sumFlux = np.nansum(imgSub * srcMask,spatialAx)
            sumErr = np.sqrt(np.nansum(varImg * srcMask,spatialAx))
            
            optSpectra[oneSrc,:] = optflux
            optSpectra_err[oneSrc,:] = np.sqrt(varFlux)
            
            sumSpectra[oneSrc,:] = sumFlux
            sumSpectra_err[oneSrc,:] = sumErr
            
            backSpectra[oneSrc,:] = np.nanmean(bkgModel * srcMask,spatialAx)
            
        
        extractDict = {} ## spectral extraction dictionary
        extractDict['t0'] = t0
        extractDict['disp indices'] = dispIndices
        extractDict['opt spec'] = optSpectra
        extractDict['opt spec err'] = optSpectra_err
        extractDict['sum spec'] = sumSpectra
        extractDict['sum spec err'] = sumSpectra_err
        extractDict['back spec'] = backSpectra
        
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
            outName = 'plots/spectra/individual_spec/{}_ind_spec_{}.pdf'.format(self.param['srcNameShort'],
                                                                                self.param['nightName'])
            fig.savefig(outName,bbox_inches='tight')
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
    
    def dyn_specFile(self,src=0):
        return "{}_src_{}.fits".format(self.dyn_specFile_prefix,src)
    
    def align_spec(self,data2D,refInd=None,diagnostics=False):
        align2D = np.zeros_like(data2D)
        nImg = data2D.shape[0]
        dispPix = self.param['dispPixels']
        
        if refInd == None:
            refInd = nImg // 2
        
        refSpec = data2D[refInd,dispPix[0]:dispPix[1]]
        waveIndices = np.arange(dispPix[1] - dispPix[0])
        
        offsetIndArr = np.zeros(nImg)
        
        for imgInd in np.arange(nImg):
            thisSpec = data2D[imgInd,dispPix[0]:dispPix[1]]
            if (imgInd > 199) & (diagnostics == True):
                doDiagnostics = True
            else:
                doDiagnostics = False
            
            offsetX, offsetInd = analysis.crosscor_offset(waveIndices,refSpec,thisSpec,Noffset=20,
                                                          diagnostics=doDiagnostics,subPixel=True,
                                                          lowPassFreq=self.param['lowPassFreqCC'],
                                                          highPassFreq=self.param['hiPassFreqCC'])
            if doDiagnostics == True:
                pdb.set_trace()
            
            align2D[imgInd,:] = analysis.roll_pad(data2D[imgInd,:],offsetInd * self.param['specShiftMultiplier'])
            offsetIndArr[imgInd] = offsetInd * self.param['specShiftMultiplier']
        
        return align2D, offsetIndArr
    
    def plot_dynamic_spec(self,src=0,saveFits=True,specAtTop=True,align=True,
                          alignDiagnostics=False):
        HDUList = fits.open(self.specFile)
        extSpec = HDUList['OPTIMAL SPEC'].data[src]
        
        nImg = extSpec.shape[0]
        
        if align == True:
            useSpec, specOffsets = self.align_spec(extSpec,diagnostics=alignDiagnostics)
        else:
            useSpec = extSpec
            specOffsets = np.zeros(nImg)
        
        avgSpec = np.nanmean(useSpec,0)
        waveIndices = HDUList['DISP INDICES'].data
        normImg = np.tile(avgSpec,[nImg,1])
        dynamicSpec = useSpec / normImg
        dynamicSpec_err = HDUList['OPT SPEC ERR'].data[src] / normImg
        
        if saveFits == True:
            dynHDU = fits.PrimaryHDU(dynamicSpec,HDUList['OPTIMAL SPEC'].header)
            dynHDU.name = 'DYNAMIC SPEC'
            dynHDU.header['ALIGNED'] = (align, 'Are the spectra shifted to align with each other?')
            
            dynHDUerr = fits.ImageHDU(dynamicSpec_err,dynHDU.header)
            dynHDUerr.name = 'DYN SPEC ERR'
            dispHDU = HDUList['DISP INDICES']
            timeHDU = HDUList['TIME']
            
            offsetHDU = fits.ImageHDU(specOffsets)
            offsetHDU.name = 'SPEC OFFSETS'
            offsetHDU.header['BUNIT'] = ('pixels','units of Spectral offsets')
            
            outHDUList = fits.HDUList([dynHDU,dynHDUerr,dispHDU,timeHDU,offsetHDU])
            outHDUList.writeto(self.dyn_specFile(src),overwrite=True)
        
        if specAtTop == True:
            fig, axArr = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': [1, 3]})
            axTop = axArr[0]
            axTop.plot(waveIndices,avgSpec)
            ax = axArr[1]
        else:
            fig, ax = plt.subplots()
        
        ax.imshow(dynamicSpec,vmin=0.95,vmax=1.05)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        ax.set_xlabel('Disp (pixels)')
        ax.set_ylabel('Time (Image #)')
        dispPix = self.param['dispPixels']
        ax.set_xlim(dispPix[0],dispPix[1])
        
        dyn_spec_name = '{}_dyn_spec_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
        fig.savefig('plots/spectra/dynamic_spectra/{}'.format(dyn_spec_name),bbox_inches='tight')
        plt.close(fig)
        
        HDUList.close()
    
    def plot_spec_offsets(self,src=0):
        if os.path.exists(self.dyn_specFile(src)) == False:
            self.plot_dynamic_spec(src=src,saveFits=True)
        HDUList = fits.open(self.dyn_specFile(src))
        time = HDUList['TIME'].data
        specOffset = HDUList['SPEC OFFSETS'].data
        plt.plot(time,specOffset)
    
    def wavebin_specFile(self,nbins=10):
        return "{}_wavebin_{}.fits".format(self.wavebin_file_prefix,nbins)
    
    def make_wavebin_series(self,specType='Optimal',src=0,nbins=10,dispIndices=None):
        if os.path.exists(self.dyn_specFile(src)) == False:
            self.plot_dynamic_spec(src=src,saveFits=True)
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
            dispSt, dispEnd = self.param['dispPixels']
        else:
            dispSt, dispEnd = dispIndices
        
        binEdges = np.array(np.linspace(dispSt,dispEnd,nbins+1),dtype=np.int)
        binStarts = binEdges[0:-1]
        binEnds = binEdges[1:]
        binIndices = np.arange(len(binStarts))
        
        binGrid = np.zeros([nTime,nbins])
        binGrid_err = np.zeros_like(binGrid)
        
        binned_disp = np.zeros(nbins)
        
        for ind, binStart, binEnd in zip(binIndices,binStarts,binEnds):
            binGrid[:,ind] = np.nansum(dynSpec[:,binStart:binEnd] * weights[:,binStart:binEnd] ,1)
            binned_disp[ind] = np.mean([binStart,binEnd])
            #binGrid_err[:,ind] = np.sqrt(np.nansum(dynSpec_err[:,binStart:binEnd]**2,1))
            binGrid[:,ind] = binGrid[:,ind] / np.nanmedian(binGrid[:,ind])
            
            #plt.errorbar(time - offset_time,binGrid[:,ind] - 0.005 * ind,fmt='o')
            #plt.errorbar(time - offset_time,binGrid[:,ind] - 0.005 * ind,
            #             yerr=binGrid_err[:,ind])
        
        outHDU = fits.PrimaryHDU(binGrid,HDUList[0].header)
        outHDU.name = 'BINNED F'
        timeHDU = HDUList['TIME']
        dispHDU = fits.ImageHDU(binned_disp,HDUList['DISP INDICES'].header)
        outHDUList = fits.HDUList([outHDU,timeHDU,dispHDU])
        outHDUList.writeto(self.wavebin_specFile(nbins),overwrite=True)
        
        HDUList.close()
        
    def plot_wavebin_series(self,nbins=10,offset=0.005,savePlot=True,yLim=None,
                            recalculate=False,dispIndices=None):
        """ Plot wavelength-binned time series """
        if (os.path.exists(self.wavebin_specFile(nbins=nbins)) == False) | (recalculate == True):
            self.make_wavebin_series(nbins=nbins,dispIndices=dispIndices)
        
        HDUList = fits.open(self.wavebin_specFile(nbins=nbins))
        time = HDUList['TIME'].data
        offset_time = np.floor(np.min(time))
        
        disp = HDUList['DISP INDICES'].data
        binGrid = HDUList['BINNED F'].data
        
        fig, ax = plt.subplots()
        for ind,oneDisp in enumerate(disp):
            ax.errorbar(time - offset_time,binGrid[:,ind] - offset * ind,fmt='o')
        
        if yLim is not None:
            ax.set_ylim(yLim[0],yLim[1])
        
        if savePlot == True:
            fig.savefig('plots/spectra/wavebin_tseries/wavebin_tser_{}.pdf'.format(self.dataFileDescrip))
            plt.close(fig)
        
        else:
            fig.show()
        
        HDUList.close()
    
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
    
    def plot_broadband_series(self,src=0,savePlot=False):
        t = self.get_broadband_series(src=src)
        offset_time = np.floor(np.min(t['time']))
        
        err_ppm = np.nanmedian(t['Norm Flux Err']) * 1e6
        print('Formal Err = {} ppm '.format(err_ppm))
        fig, ax = plt.subplots()
        
        ax.plot(t['time'] - offset_time,t['Norm Flux'])
        ax.set_xlabel("Time - {} (days)".format(offset_time))
        ax.set_ylabel("Normalized Flux")
        if savePlot == True:
            bb_series_name = '{}_bb_series_{}.pdf'.format(self.param['srcNameShort'],self.param['nightName'])
            outName = 'plots/spectra/broadband_series/{}'.format(bb_series_name)
            fig.savefig(outName)
        else:
            fig.show()
        

class batch_spec(phot_pipeline.batchPhot):
    def __init__(self,batchFile='parameters/spec_params/example_batch_spec_parameters.yaml'):
        self.alreadyLists = {'starPositions': 1,'bkgRegionsX': 2, 'bkgRegionsY': 2,
                             'dispPixels': 1, 'excludeList': 1}
        self.general_init(batchFile=batchFile)
    
    def make_pipe_obj(self,directParam):
        """
        Make a spectroscopy pipeline object that will be executed in batch
        """
        return spec(directParam=directParam)
    
    def run_all(self,useMultiprocessing=False):
        self.batch_run('do_extraction')
        self.batch_run('plot_dynamic_spec',saveFits=True)
    
    def plot_all(self):
        self.batch_run('plot_one_spec',savePlot=True)
    
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

comparisonFileNames = glob.glob('tser_data/spec/spec_o9*.fits')

def compare_spectra(fileNames=comparisonFileNames,specType='Optimal',showPlot=False):
    fig, ax = plt.subplots()
    for oneFile in fileNames:
        x, y, yerr = get_spectrum(oneFile,specType=specType)
        head = fits.getheader(oneFile)
        ax.plot(x,y,label=head['SRCNAME'])
    ax.legend()
    if showPlot == True:
        plt.show()
    else:
        fig.savefig('plots/spectra/comparison_spec/comparison_spec.pdf')
        plt.close(fig)
        