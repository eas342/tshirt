import phot_pipeline
import spec_pipeline
from astropy.io import fits, ascii
import os
import matplotlib.pyplot as plt
import pdb
import numpy as np

def test_binning():
    """ Test the binning function"""
    x = np.linspace(0,10,1024)
    y = np.random.randn(1024)
    
    plt.plot(x,y,'.')
    xBin, yBin, yErr = phot_pipeline.do_binning(x,y)
    
    stdevOrig = np.std(y)
    print('Stdev orig = {}, theoretically 1.0'.format(stdevOrig))
    
    ptsPerBin = np.float(len(x)) / np.float(len(xBin))
    print("pts per bin = {}".format(ptsPerBin))
    expectedStd = 1./np.sqrt(ptsPerBin)
    print("expected stdev of binned = {}".format(expectedStd))
    print("measured stdev of binned = {}".format(np.std(yBin)))
    print("median errorbar = {}".format(np.median(yErr)))
    
    plt.errorbar(xBin,yBin,yErr,fmt='o')
    plt.show()
    