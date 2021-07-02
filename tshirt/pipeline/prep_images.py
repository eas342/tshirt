import glob
try:
    from ccdproc import Combiner, CCDData, ccd_process, gain_correct
    import ccdproc
except ImportError as err1:
    print("Could not import ccdproc, so image processing may not work")
import yaml
import os
from astropy.io import fits
import astropy.units as u
import numpy as np
import pdb
import warnings
from scipy.interpolate import interp1d
import tqdm


defaultParamFile = 'parameters/reduction_parameters/example_reduction_parameters.yaml'
class prep():
    """ Class for reducing images
    Parameters are located in parameters/reduction_parameters
     """
    def __init__(self,paramFile=defaultParamFile,testMode=False,rawIndex=0):
        with open(paramFile) as paramFileIO:
            self.pipePrefs = yaml.safe_load(paramFileIO)
        
        
        defaultParams = {'biasFiles': 'zero*.fits', ## Bias files
                         'flatFiles': 'flat*.fits', ## Flat field files
                         'sciFiles': 'k1255*.fits', ## Science files
                         'nSkip': 2, ## number of files to skip at the beginning
                         'doBias': True, ## Calculate and apply a bias correction?
                         'doFlat': True, ## Calculate and apply a flat field correction
                         'doBadPxMask': False, ## Appply a bad pixel mask?
                         'doFlatBias': False, ## use a specific bias or dark for the flat field
                         'darksForFlat': None, ## dark frames for master flat frame
                         'gainKeyword': 'GAIN1', ## Calculate and apply a flat correction?
                         'gainValue': None,## manually specify the gain, if not in header
                         'procName': 'proc', ## directory name for processed files
                         'doNonLin': False, ## apply nonlinearity correction?
                         'nonLinFunction': None, ## non-linearity function
                         'sciExtension': None, ## extension for science data
                         'sciExcludeList': None, ## list of files to exclude for science data
                         'fixWindow': False, ## fix the window between bias, flat & science?
                         'fixPix': False, ## fix the bad pixels with interpolation?
                         'combinerFunc': 'average' ## how to combine calibration files?
                     } 
        
        for oneKey in defaultParams.keys():
            if oneKey not in self.pipePrefs:
                self.pipePrefs[oneKey] = defaultParams[oneKey]
        
        rawDirInput = self.pipePrefs['rawDir']
        if type(rawDirInput) == list:
            ## If the parameter file is list, choose one of those indices
            self.rawDir = rawDirInput[rawIndex]
            self.nRawDirs = len(rawDirInput)
        elif type(rawDirInput) == str:
            self.rawDir = rawDirInput
            self.nRawDirs = 1
        else:
            print('Unrecognized input file directory/list')
        
        self.testMode = testMode
        if testMode == True:
            self.procDir = os.path.join(self.rawDir,'test_proc')
        else:
            self.procDir = os.path.join(self.rawDir,self.pipePrefs['procName'])
        
    
    def makeMasterCals(self):
        """ Makes the master calibration files """
        
        allCals = []
        if self.pipePrefs['doBias'] == True:
            allCals.append('biasFiles')
        
        if self.pipePrefs['doFlat'] == True:
            allCals.append('flatFiles')
        
        if self.pipePrefs['doFlatBias'] == True:
            allCals.append('darksForFlat')
        
        for oneCal in allCals:
            fileSearchInfo = self.pipePrefs[oneCal]
            fileL = self.get_fileL(fileSearchInfo)
            
            if self.testMode == True:
                fileL = fileL[0:4]
            
            ccdList = []
            for oneFile in fileL:
                head, dataCCD = self.getData(oneFile)
                ccdList.append(dataCCD)
            
            combiner = Combiner(ccdList)
            
            combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
            combinerFunc = self.pipePrefs['combinerFunc']
            if combinerFunc == 'average':
                combined_avg = combiner.average_combine()
            elif combinerFunc == 'median':
                combined_avg = combiner.median_combine()
            else:
                raise Exception("Unrecognized combiner function {}".format(self.combiner_func))
            
            
            comb_gained = gain_correct(combined_avg,self.get_gain(head) * u.electron/u.adu)
            
            hdu = fits.PrimaryHDU(comb_gained,head)
            HDUList = fits.HDUList([hdu])
            HDUList[0].header['IMGAVG'] = ('T','Image is averaged')
            HDUList[0].header['GAINCOR'] = ('T','Gain correction applied (units are e)')
            HDUList[0].data = combined_avg
            
            if os.path.exists(self.procDir) == False:
                os.mkdir(self.procDir)
            
            if oneCal == 'biasFiles':
                outName = 'zero'
            elif oneCal == 'darksForFlat':
                outName = 'dark_for_flat'
            else:
                outName = 'flat'
            HDUList.writeto(os.path.join(self.procDir,'master_'+outName+'.fits'),overwrite=True)
    
    def get_gain(self,head):
        if self.pipePrefs['gainKeyword'] is not None:
            gain = float(head[self.pipePrefs['gainKeyword']])
        elif self.pipePrefs['gainValue'] is not None:
            gain = float(self.pipePrefs['gainValue'])
        else:
            gain = 1.0
        return gain
    
    def get_fileL(self,fileSearchInfo,searchType='generic'):
        """
        Search for a list of files
        Tests out if the user put in a string w/ wildcard or a list of files
        """
        if type(fileSearchInfo) == list:
            fileL = []
            for oneFile in fileSearchInfo:
                fileL.append(os.path.join(self.rawDir,oneFile))
        else:
            fileL = np.sort(glob.glob(os.path.join(self.rawDir,fileSearchInfo)))
            
        if (self.pipePrefs['sciExcludeList'] is not None) & (searchType == 'science'):
            outList = []
            
            for oneFile in fileL:
                if os.path.basename(oneFile) not in self.pipePrefs['sciExcludeList']:
                    outList.append(oneFile)
        else:
            outList = fileL
        return outList
    
    
    def procSciFiles(self):
        """ Process the science images """
        fileL = self.get_fileL(self.pipePrefs['sciFiles'],searchType='science')
        if self.testMode == True:
            fileL = fileL[0:4]
        
        if self.pipePrefs['doBias'] == True:
            hbias, bias = self.getData(os.path.join(self.procDir,'master_zero.fits'))
        else:
            hbias, bias = None, None
        
        if self.pipePrefs['doFlat'] == True:
            hflat, flat = self.getData(os.path.join(self.procDir,'master_flat.fits'))
            
            if self.pipePrefs['doFlatBias'] == True:
                hFlatDark, flatDark = self.getData(os.path.join(self.procDir,'master_dark_for_flat.fits'))
                flat = ccdproc.subtract_bias(flat,flatDark)
            
        else:
            hflat, flat = None, None
        
        if self.pipePrefs['doBadPxMask'] == True:
            badPx = fits.getdata(os.path.join(self.procDir,'master_badpx_mask.fits'))
        else:
            hbadPx, badPx = None, None
        
        for ind in tqdm.tqdm(np.arange(len(fileL))):
            oneFile = fileL[ind]
            head, dataCCD = self.getData(oneFile)
            
            sciHead = fits.getheader(oneFile,ext=self.pipePrefs['sciExtension'])
            if ('CCDSEC' in sciHead) & (self.pipePrefs['fixWindow'] == True):
                yFix =  np.array(sciHead['CCDSEC'].split(',')[1].split(']')[0].split(':'),dtype=np.int) - 1
                
                if flat is not None:
                    useFlat = flat[yFix[0]:yFix[1]+1]
                    if ind == 0:
                        flatSave = fits.PrimaryHDU(useFlat)
                        flatSave.writeto('diagnostics/trimmed_flat/trimmed_flat.fits',overwrite=True)
                if bias is not None:
                    useBias = bias[yFix[0]:yFix[1]+1]#ccdproc.trim_image(bias,sciHead['CCDSEC'])
                    
            else:
                useFlat = flat
                useBias = bias
            
            
            nccd = ccd_process(dataCCD,gain=self.get_gain(head) * u.electron/u.adu,
                               master_flat=useFlat,
                               master_bias=useBias,
                               bad_pixel_mask=badPx)
            
            if nccd.mask is not None:
                nccd.data[nccd.mask] = np.nan
            
            if self.pipePrefs['fixPix'] == True:
                nccd.data = self.fix_pix_line(nccd.data)
            
            head['ZEROFILE'] = 'master_zero.fits'
            head['FLATFILE'] = 'master_flat.fits'
            head['BADPXFIL'] = 'master_badpx_mask.fits'
            head['GAINCOR'] = ('T','Gain correction applied (units are e)')
            head['BUNIT'] = ('electron','Physical unit of array values')
            
            hdu = fits.PrimaryHDU(data=nccd,header=head)
            HDUList = fits.HDUList([hdu])
            newFile = os.path.basename(oneFile)
            HDUList.writeto(os.path.join(self.procDir,newFile),overwrite=True)
        
    def fix_pix(self,x,y):
        goodP = np.isfinite(y)
        if np.sum(goodP) > 5:
            y_model = interp1d(x[goodP],y[goodP],fill_value="extrapolate")
            y_out = y_model(x)
        else:
            y_out = y
        
        return y_out
    
    def fix_pix_line(self,img,direction='row'):
        """ 
        Fix pixels
        """
        if direction == 'row':
            nY = img.shape[0]
            x = np.arange(img.shape[1])
            for row in np.arange(nY):
                img[row,:] = self.fix_pix(x,img[row,:])
        else:
            raise NotImplementedError
        return img
    
    def check_if_nonlin_needed(self,head):
        """
        Check if non-linearity correction should be applied
        """
        ## Is the nonlinearity correction specified?
        
        if self.pipePrefs['doNonLin'] == True:
            if 'LINCOR' in head:
                if head['LINCOR'] == True:
                    return False
                else:
                    return True ## run if lincor hasn't been applied
            else:
                return True ## run if LINCOR not in header
        else:
            return False
    
    
    def getData(self,fileName):
        """ Gets the data and converts to CCDData type"""
        HDUList = fits.open(fileName)
        if self.pipePrefs['sciExtension'] is None:
            sciExtension = 0
        else:
            if ('master_flat' in fileName) | ('master_zero' in fileName):
                sciExtension = 0
            else:
                sciExtension = self.pipePrefs['sciExtension']
        
        if sciExtension >= len(HDUList):
            print('No extension {} for {}. Trying 0'.format(sciExtension,fileName))
            sciExtension = 1
        
        data = HDUList[sciExtension].data
        head = HDUList[0].header
        HDUList.close()
        
        if 'GAINCOR' in head:
            if head['GAINCOR'] == 'T':
                outUnit = u.electron
            else:
                outUnit = u.adu
        else:
            outUnit = u.adu
        
        if self.check_if_nonlin_needed(head) == True:
            if 'LBT LUCI' in self.pipePrefs['nonLinFunction']:
                if 'NDIT' not in head:
                    print("No NDIT found for {} so no non-linearity correction applied".format(fileName))
                    data = data
                else:
                    if self.pipePrefs['nonLinFunction'] == 'LBT LUCI2':
                        data = lbt_luci2_lincor(data,dataUnit=outUnit,ndit=head['NDIT'])
                    elif self.pipePrefs['nonLinFunction'] == 'LBT LUCI2 OLD':
                        data = lbt_luci2_lincor(data,dataUnit=outUnit,ndit=head['NDIT'],k2=4.155e-6)
                    else:
                        raise Exception("Unrecognized non-linearity function {}".format(self.pipePrefs['nonLinFunction']))
            else:
                raise Exception("Unrecognized non-linearity function {}".format(self.pipePrefs['nonLinFunction']))
            head['LINCOR'] = (True, "Is a non-linearity correction applied?")
            head['LINCFUNC'] = (self.pipePrefs['nonLinFunction'], "Name of non-linearity function applied")
        else:
            head['LINCOR'] = (False, "Is a non-linearity correction applied?")
        
        try:
            outData = CCDData(data,unit=outUnit)
        except TypeError as err1:
            pdb.set_trace()
        
        return head, outData

def lbt_luci2_lincor(img,dataUnit=u.adu,ndit=1.0,k2=2.767e-6):
    """
    LUCI2 linearity correction from 
    https://sites.google.com/a/lbto.org/luci/observing/calibrations/calibration-details
    
    Input image should be in ADU
    
    Parameters
    ----------
    img: numpy array
        An input image to do linearity correction on
    
    dataUnit: astropy unit
        Unit of the image, such as :code:`astropy.units.adu`
    
    ndit: float
        The number of the detection integration time in LUCI2's readout system
        This has to be updated for the science observations in question
    
    k2: float
        Quadratic coefficient in non-linearity correction
    
    Returns
    -------
    ADUlin * ndit: numpy array
        A new image that has been linearity-corrected
    
    """
    if dataUnit == u.adu:
        imgUse = img
    else:
        raise Exception("Unit {} not the right unit for this nonlin correction".format(datUnit))
    imgUse = imgUse / ndit
    ADUlin=imgUse + k2 * (imgUse)**2
    return ADUlin * ndit

def reduce_all(testMode=False):
    """ Reduce all files listed in reduction parameters """
    pipeObj = prep()
    for rawIndex in range(pipeObj.nRawDirs):
        pipeObj = prep(rawIndex=rawIndex,testMode=testMode)
        print("Making Master Cals")
        pipeObj.makeMasterCals()
        print("Processing Science Files")
        pipeObj.procSciFiles()
        

if __name__ == "__main__":
    reduce_all()
    
