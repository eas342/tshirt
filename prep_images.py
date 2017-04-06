import glob
from ccdproc import Combiner, CCDData, ccd_process, gain_correct
import yaml
import os
from astropy.io import fits
import astropy.units as u
import numpy as np
import pdb

pipePrefs = yaml.load(open('parameters/reduction_parameters.yaml'))
rawDir = pipePrefs['rawDir']
procDir = os.path.join(rawDir,'proc')

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

def makeMasterCals(testMode=False):
    """ Makes the master calibration files """
    
    for oneCal in ['biasFiles','flatFiles']:
        
        fileL = glob.glob(os.path.join(rawDir,pipePrefs[oneCal]))
        if testMode == True:
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
        
        if os.path.exists(procDir) == False:
            os.mkdir(procDir)
        
        if oneCal == 'biasFiles':
            outName = 'zero'
        else:
            outName = 'flat'
        HDUList.writeto(os.path.join(procDir,'master_'+outName+'.fits'),overwrite=True)

def procSciFiles():
    """ Process the science images """
    fileL = glob.glob(os.path.join(rawDir,pipePrefs['sciFiles']))
    if testMode == True:
        fileL = fileL[0:4]
    
    hbias, bias = getData(os.path.join(procDir,'master_zero.fits'))
    hflat, flat = getData(os.path.join(procDir,'master_flat.fits'))
    
    for oneFile in fileL:
        head, dataCCD = getData(oneFile)
        nccd = ccd_process(dataCCD,gain=float(head['GAIN1']) * u.electron/u.adu,
                           master_flat=flat,
                           master_bias=bias)
        hdu = fits.PrimaryHDU(nccd)
        HDUList = fits.HDUList([hdu])
        HDUList[0].header = head
        HDUList[0].data = nccd
        newFile = os.path.basename(oneFile)
        HDUList.writeto(os.path.join(procDir,newFile),overwrite=False)
        
        

if __name__ == "__main__":
    print("Making Master Cals")
    makeMasterCals()
    print("Processing Science Files")
    procSciFiles()
    
