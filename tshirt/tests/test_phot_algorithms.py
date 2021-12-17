import unittest
from tshirt.pipeline import phot_pipeline, spec_pipeline
from tshirt.pipeline.instrument_specific import rowamp_sub
from astropy.io import fits,ascii
from pkg_resources import resource_filename
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

if __name__ == '__main__':
    unittest.main()
    