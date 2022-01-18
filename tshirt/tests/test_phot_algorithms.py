import unittest
from tshirt.pipeline import phot_pipeline, spec_pipeline
from tshirt.pipeline.instrument_specific import rowamp_sub
from astropy.io import fits,ascii
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
from copy import deepcopy
from tshirt.pipeline import sim_data
import os
import numpy as np
from tshirt.pipeline.utils import get_baseDir

class driftPhot(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        testPhotParams = 'parameters/phot_params/test_params/test_drifting_psf.yaml'
        testParamPath = phot_pipeline.resource_filename('tshirt',testPhotParams)
        self.phot = phot_pipeline.phot(testParamPath)
        if self.phot.nImg == 0:
            sim_data.sim_phot_w_large_shift()
            self.phot = phot_pipeline.phot(testParamPath)
        
        drift_file = os.path.join(get_baseDir(),
                                  'example_tshirt_data',
                                  'sim_data','drift_phot',
                                  'drift.csv')
        self.drift_tab = ascii.read(drift_file)
    
    def test_no_default_drift(self):
        phot = phot_pipeline.phot()
        self.assertTrue(np.allclose(phot.drift_dat['dx'],0.0))
        self.assertTrue(np.allclose(phot.drift_dat['dy'],0.0))
    
    def test_centering(self):
        self.phot.get_allimg_cen(recenter=True,useMultiprocessing=False)
        pos = self.phot.cenArr
        xArr = pos[:,0,0]
        yArr = pos[:,0,1]
        self.assertTrue(np.max(np.abs(xArr - self.drift_tab['x truth'])) < 2.)
        self.assertTrue(np.max(np.abs(yArr - self.drift_tab['y truth'])) < 2.)
        
    
    def test_phot_result(self):
        self.phot.get_allimg_cen(recenter=True,useMultiprocessing=True)
        self.phot.do_phot(useMultiprocessing=True)
        t1 = self.phot.print_phot_statistics(returnOnly=True)
        ## make sure error is 
        self.assertTrue(t1['Stdev (%)'][0] < 20.)

class rowAmpBacksub(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        testPhotParams = 'parameters/phot_params/test_params/test_rowamp_sub.yaml'
        testParamPath = phot_pipeline.resource_filename('tshirt',testPhotParams)
        self.phot = phot_pipeline.phot(testParamPath)
    
    def test_backsub(self):
        img, head = self.phot.get_default_im()
        outimg, outmodel = rowamp_sub.do_backsub(img,self.phot,saveDiagnostics=True,evenOdd=False)
        diag_dir = os.path.join(get_baseDir(),'diagnostics','rowamp_sub')
        out_path = os.path.join(diag_dir,"{}_subtracted.fits".format(self.phot.dataFileDescrip))
        self.assertTrue(os.path.exists(out_path))
    
    def test_backsub_with_even_odd(self):
        param = deepcopy(self.phot.param)
        param['srcNameShort'] = 'test_rowamp_even_odd'
        newPhot = phot_pipeline.phot(directParam=param)
        img, head = newPhot.get_default_im()
        outimg, outmodel = rowamp_sub.do_backsub(img,newPhot,saveDiagnostics=True,evenOdd=True)
        diag_dir = os.path.join(get_baseDir(),'diagnostics','rowamp_sub')
        out_path = os.path.join(diag_dir,"{}_subtracted.fits".format(newPhot.dataFileDescrip))
        self.assertTrue(os.path.exists(out_path))
    
    def test_phot_result(self):
        self.phot.do_phot(useMultiprocessing=False)
        ## make sure the phot looks good
        #self.assertTrue(t1['Stdev (%)'][0] < 20.)
    
    def sim_data(self):
        """
        Simulate row-to-row variability, a mask and a source
        """
        dimenx=32
        dimeny=256
        cen = [16,128]
        
        x_1d = np.arange(dimenx)
        y_1d = np.arange(dimeny)
        x, y = np.meshgrid(x_1d, y_1d) # get 2D variables instead of 1D
        
        gauss2d = sim_data.gauss_2d(x,y,cen[0],cen[1],
                                    sigx=2.0,sigy=2.0,
                                    norm=1e3)
        
        r = np.sqrt((cen[0] - x)**2 + (cen[1]- y)**2)
        
        ## mask all points well away from the source
        bkgmask = (r > 10.)
        
        np.random.seed(0)
        simnoise_1d = np.random.randn(dimeny)
        simnoise_2d = np.tile(simnoise_1d,[dimenx,1]).T
        
        simdata = simnoise_2d + gauss2d
        
        
        outDict = {}
        outDict['simdata'] = simdata
        outDict['bkgmask'] = bkgmask
        outDict['gauss2d'] = gauss2d
        
        return outDict
    
    def show_images(self,imgDict=None):
        """
        Show the simulation, if asked.
        """
        
        if imgDict == None:
            imgDict = self.sim_data()
        keys = imgDict.keys()
        fig, axArr = plt.subplots(1,len(keys))
        for ind,oneImg in enumerate(keys):
            axArr[ind].imshow(imgDict[oneImg])
            axArr[ind].set_title(oneImg)
        
        fig.show()
    
    def test_masked_roeba(self,showImg=False):
        simDict = self.sim_data()
        outimg, modelimg = rowamp_sub.do_backsub(simDict['simdata'],
                                                 amplifiers=1,
                                                 backgMask=simDict['bkgmask'])
        simDict['model'] = modelimg
        simDict['outimg'] = outimg
        simDict['resid'] = simDict['gauss2d'] - simDict['outimg']
        if showImg == True:
            self.show_images(simDict)
        self.assertTrue(np.allclose(simDict['gauss2d'],simDict['outimg']))

if __name__ == '__main__':
    unittest.main()
    