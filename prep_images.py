import glob
from ccdproc import Combiner, CCDData, ccd_process, gain_correct
import yaml
import os
from astropy.io import fits
import astropy.units as u
import numpy as np
import pdb

class prep():
    """ Class for reducing images
    Parameters are located in parameters/reduction_parameters
     """
    def __init__(self,testMode=False,rawIndex=0):
        self.pipePrefs = yaml.load(open('parameters/reduction_parameters.yaml'))
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
    
        for oneCal in ['biasFiles','flatFiles']:
        
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
            
            comb_gained = gain_correct(combined_avg,float(head['GAIN1']) * u.electron/u.adu)
            
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

    def procSciFiles(self):
        """ Process the science images """
        fileL = glob.glob(os.path.join(self.rawDir,self.pipePrefs['sciFiles']))
        if self.testMode == True:
            fileL = fileL[0:4]
        
        hbias, bias = getData(os.path.join(self.procDir,'master_zero.fits'))
        hflat, flat = getData(os.path.join(self.procDir,'master_flat.fits'))
        
        for oneFile in fileL:
            head, dataCCD = getData(oneFile)
            nccd = ccd_process(dataCCD,gain=float(head['GAIN1']) * u.electron/u.adu,
                               master_flat=flat,
                               master_bias=bias)
            head['ZEROFILE'] = 'master_zero.fits'
            head['FLATFILE'] = 'master_flat.fits'
            head['GAINCOR'] = ('T','Gain correction applied (units are e)')
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
    
