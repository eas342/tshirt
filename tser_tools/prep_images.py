import glob
from ccdproc import Combiner, CCDData, ccd_process, gain_correct
import yaml
import os
from astropy.io import fits
import astropy.units as u
import numpy as np
import pdb

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
                         'doFlat': True,
                         'gainKeyword': 'GAIN1', ## Calculate and apply a flat correction?
                         'gainValue': None} ## manually specify the gain, if not in header
        
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
            self.procDir = os.path.join(self.rawDir,'proc')
        
    
    def makeMasterCals(self):
        """ Makes the master calibration files """
        
        allCals = []
        if self.pipePrefs['doBias'] == True:
            allCals.append('biasFiles')
        
        if self.pipePrefs['doFlat'] == True:
            allCals.append('flatFiles')
        
        for oneCal in allCals:
        
            fileL = glob.glob(os.path.join(self.rawDir,self.pipePrefs[oneCal]))
            if self.testMode == True:
                fileL = fileL[0:4]
            
            ccdList = []
            for oneFile in fileL:
                head, dataCCD = getData(oneFile)
                ccdList.append(dataCCD)
            
            combiner = Combiner(ccdList)
            
            combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
            combined_avg = combiner.average_combine()
            
            comb_gained = gain_correct(combined_avg,self.get_gain(head) * u.electron/u.adu)
            
            hdu = fits.PrimaryHDU(comb_gained)
            HDUList = fits.HDUList([hdu])
            HDUList[0].header['IMGAVG'] = ('T','Image is averaged')
            HDUList[0].header['GAINCOR'] = ('T','Gain correction applied (units are e)')
            HDUList[0].data = combined_avg
            
            if os.path.exists(self.procDir) == False:
                os.mkdir(self.procDir)
            
            if oneCal == 'biasFiles':
                outName = 'zero'
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

    def procSciFiles(self):
        """ Process the science images """
        fileL = glob.glob(os.path.join(self.rawDir,self.pipePrefs['sciFiles']))
        if self.testMode == True:
            fileL = fileL[0:4]
        
        if self.pipePrefs['doBias'] == True:
            hbias, bias = getData(os.path.join(self.procDir,'master_zero.fits'))
        else:
            hbias, bias = None, None
        
        if self.pipePrefs['doFlat'] == True:
            hflat, flat = getData(os.path.join(self.procDir,'master_flat.fits'))
        else:
            hflat, flat = None, None
        
        for oneFile in fileL:
            head, dataCCD = getData(oneFile)
            nccd = ccd_process(dataCCD,gain=self.get_gain(head) * u.electron/u.adu,
                               master_flat=flat,
                               master_bias=bias)
            head['ZEROFILE'] = 'master_zero.fits'
            head['FLATFILE'] = 'master_flat.fits'
            head['GAINCOR'] = ('T','Gain correction applied (units are e)')
            head['BUNIT'] = ('electron','Physical unit of array values')
            hdu = fits.PrimaryHDU(data=nccd,header=head)
            HDUList = fits.HDUList([hdu])
            newFile = os.path.basename(oneFile)
            HDUList.writeto(os.path.join(self.procDir,newFile),overwrite=True)
    

def getData(fileName):
    """ Gets the data and converts to CCDData type"""
    HDUList = fits.open(fileName)
    data = HDUList[0].data
    head = HDUList[0].header
    HDUList.close()
    
    if 'GAINCOR' in head:
        if head['GAINCOR'] == 'T':
            outUnit = u.electron
        else:
            outUnit = u.adu
    else:
        outUnit = u.adu
    
    outData = CCDData(data,unit=outUnit)
    return head, outData

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
    
