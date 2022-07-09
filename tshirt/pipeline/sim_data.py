import photutils
from astropy.io import fits, ascii
import sys
import os
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
import glob
from photutils import CircularAperture, CircularAnnulus
from photutils import RectangularAperture
from photutils import aperture_photometry
if photutils.__version__ > "1.0":
    from photutils.centroids import centroid_2dg
else:
    from photutils import centroid_2dg
import photutils
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
import time
import logging
import urllib
import tqdm
from .phot_pipeline import get_baseDir

def gauss_2d(x,y,x0,y0,sigx=1.0,sigy=1.0,norm=1.0):
    """
    A 2D Gaussian function
    """
    arg = -1.0 * ((x - x0)**2/(2. * sigx**2) + 
                  (y - y0)**2/(2. * sigy**2))
    ## only evaluate exponential where it will avoid
    ## undeflow errors for tiny results
    high_pts = arg >= -15.
    z = np.zeros_like(x,dtype=float)
    z[high_pts] = np.exp(arg[high_pts]) /(2. * np.pi * sigx * sigy)
    
    return z * norm

def make_gauss_star(dimen=30,cen=[15,15],flux=1.0):
    x = np.arange(dimen)
    y = np.arange(dimen)
    #x = np.linspace(-halfdimen, halfdimen,)
    #y = np.linspace(-halfdimen, halfdimen)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z = gauss_2d(x, y, cen[0],cen[1],norm=flux)
    return z

def sim_phot_w_large_shift():
    
    nImg = 10
    dimen = 30
    xcen = np.linspace(4,25,nImg)
    ycen = np.linspace(25,4,nImg)
    
    time_start = Time('2020-05-04T00:00:00.0',format='fits')
    time_obs = time_start + np.arange(nImg) * 10.0 * u.second
    
    outDir = os.path.join(get_baseDir(),
                          'example_tshirt_data',
                          'sim_data','drift_phot')
    
    np.random.seed(0)
    fileNames = []
    for ind in np.arange(nImg):
        z = make_gauss_star(dimen=dimen,cen=[xcen[ind],ycen[ind]],flux=100.0)
        noise = np.random.randn(dimen,dimen) * 1.0
        simImg = z + noise
        
        primHDU = fits.PrimaryHDU(simImg)
        primHDU.header['DATE-OBS'] = (time_obs[ind].fits, "Int Start Time")
        
        outName = 'sim_phot_{:03d}.fits'.format(ind)
        fileNames.append(outName)
        primHDU.writeto(os.path.join(outDir,outName),overwrite=True)
    
    dr_dat = Table()
    dr_dat['Index'] = np.arange(nImg)
    dr_dat['File'] = fileNames
    dr_dat['dx'] = xcen - np.mean(xcen) + np.random.randn(nImg) * 1.
    dr_dat['dy'] = ycen - np.mean(ycen) + np.random.randn(nImg) * 1.
    dr_dat['x truth'] = xcen
    dr_dat['y truth'] = ycen
    dr_path = os.path.join(outDir,'drift.csv')
    dr_dat.write(dr_path,overwrite=True)
    