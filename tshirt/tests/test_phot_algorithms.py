import unittest
from tshirt.pipeline import phot_pipeline, spec_pipeline
from astropy.io import fits,ascii
from pkg_resources import resource_filename
from copy import deepcopy
from tshirt.pipeline import sim_data
import os
import numpy as np

class driftPhot(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        testPhotParams = 'parameters/phot_params/test_params/test_drifting_psf.yaml'
        testParamPath = phot_pipeline.resource_filename('tshirt',testPhotParams)
        self.phot = phot_pipeline.phot(testParamPath)
        if self.phot.nImg == 0:
            sim_data.sim_phot_w_large_shift()
            self.phot = phot_pipeline.phot(testParamPath)
        
        drift_file = os.path.join(os.environ["TSHIRT_DATA"],
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
        
if __name__ == '__main__':
    unittest.main()