from pipeline import phot_pipeline
from pipeline import spec_pipeline
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
    
def test_allan_variance(doYerr=True):
    """ Test the binning function"""
    yMultiplier = 500.
    x = np.linspace(0,100,2048)
    y = np.random.randn(2048) * yMultiplier
    if doYerr == True:
        yerr = np.ones_like(x) * yMultiplier
    else:
        yerr = None
    
    phot_pipeline.allan_variance(x,y,yerr)
    
def test_poly_sub():
    phot = phot_pipeline.phot(paramFile='parameters/phot_params/test_parameters/phot_param_k2_22_colrow.yaml')
    phot.param['diagnosticMode'] = True
    phot.do_phot()

def compare_colrow_and_annulus_backsub(recalculate=False):
    descriptions = ['Background Annulus','Col-Row Sub']
    
    fig, ax = plt.subplots()
    for ind,oneName in enumerate(['phot_param_k2_22_annulus.yaml','phot_param_k2_22_colrow.yaml']):
        path = os.path.join('parameters','phot_params','test_parameters',oneName)
        phot = phot_pipeline.phot(paramFile=path)
        if (os.path.exists(phot.photFile) == False) | (recalculate == True):
            phot.do_phot(useMultiprocessing=True)
        
        print("***************************")
        print(descriptions[ind])
        print("***************************")
        stats = phot.print_phot_statistics(refCorrect=False)
        
        HDUList = fits.open(phot.photFile)
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        t = jdArr - np.round(np.min(jdArr))
        jdHDU = HDUList['TIME']
        
        backData = HDUList['BACKG PHOT'].data
        
        linestyles=['-','-.']
        for oneSrc in np.arange(phot.nsrc):
            thisLabel = "{} src {}".format(descriptions[ind],oneSrc)
            ax.plot(t,backData[:,oneSrc],label=thisLabel,linestyle=linestyles[oneSrc])
        
        HDUList.close()
    ax.legend()
    ax.set_ylabel("Backg Flux")
    fig.show()