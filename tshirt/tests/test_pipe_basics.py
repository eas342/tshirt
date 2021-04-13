import unittest
from tshirt.pipeline import phot_pipeline, spec_pipeline
from astropy.io import fits
from pkg_resources import resource_filename
from copy import deepcopy

class BasicPhot(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        testPhotParams = 'parameters/phot_params/test_params/phot_param_k2_22_annulus.yaml'
        examParamPath = phot_pipeline.resource_filename('tshirt',testPhotParams)
        self.phot = phot_pipeline.phot(examParamPath)
    
    def test_plot_starChoices(self):
        phot = phot_pipeline.phot()
        phot.showStarChoices(showPlot=False,showAps=True)
    
    def test_phot_extract_single_core(self):
        self.phot.do_phot(useMultiprocessing=True)
        
    def test_phot_extract_single_core(self):
        self.phot.do_phot(useMultiprocessing=False)
        
    def test_phot_w_other_backsub(self):
        """
        Test if photometry works with median background subtraction
        """
        param = deepcopy(self.phot.param)
        for oneMethod in ['median','rowcol','robust mean']:
            param['bkgMethod'] = oneMethod
        phot = phot_pipeline.phot(directParam=param)
        phot.do_phot()
        
    # def test_simple
    #
    # def test_upper(self):
    #     print("Undefined variable={}".format(undefined_variable))
    #     self.assertEqual('foo'.upper(), 'FOO')
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

simpleDict = {'whale': 'big','ant': 'small'}
head = fits.Header(simpleDict)


class BasicSpec(unittest.TestCase):
    
    def test_initialization(self):
        spec = spec_pipeline.spec()
    
    def test_basic_spec(self):
        paramFile = 'parameters/spec_params/test_params/basic_hst_spec.yaml'
        exampleParamPath = resource_filename('tshirt',paramFile)
        spec = spec_pipeline.spec(exampleParamPath)
        spec.do_extraction()
        

class ExistsAndValue(unittest.TestCase):

    def test_if_true(self):
        self.assertTrue(phot_pipeline.exists_and_equal(simpleDict,'whale','big'))
        self.assertTrue(phot_pipeline.exists_and_equal(head,'whale','big'))
    
    def test_if_false(self):
        self.assertFalse(phot_pipeline.exists_and_equal(simpleDict,'whale','small'))
        self.assertFalse(phot_pipeline.exists_and_equal(head,'whale','small'))        
    
    def test_if_not_in(self):
        self.assertFalse(phot_pipeline.exists_and_equal(simpleDict,'microbe','small'))
        self.assertFalse(phot_pipeline.exists_and_equal(head,'microbe','small'))


if __name__ == '__main__':
    unittest.main()
