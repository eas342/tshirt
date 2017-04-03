import glob
from ccdproc import Combiner, CCDData
import yaml
import os
from astropy.io import fits
import astropy.units as u
import numpy as np

rawDir = '/data1/tso_analysis/kic1255/UT2016_06_12/'
procDir = '/data1/tso_analysis/kic1255/UT2016_06_12_proc/'

fileL = glob.glob(rawDir+'zero*.fits')

ccdList = []
for oneFile in fileL[0:2]:
    HDUList = fits.open(oneFile)
    data = HDUList[0].data
    head = HDUList[0].header
    HDUList.close()
    ccdList.append(CCDData(data,unit=u.adu))

combiner = Combiner(ccdList)

combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
combined_avg = combiner.average_combine()

hdu = fits.PrimaryHDU(combined_avg)
HDUList = fits.HDUList([hdu])
HDUList[0].header['IMGAVG'] = ('T','Image is averaged')

HDUList[0].data = combined_avg


if os.path.exists(procDir) == False:
    os.mkdir(procDir)

HDUList.writeto(os.path.join(procDir,'master_zero.fits'),overwrite=True)
