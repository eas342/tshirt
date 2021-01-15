import unittest
from tshirt.pipeline import phot_pipeline
from astropy.io import fits

class TestStringMethods(unittest.TestCase):

    def test_initialization(self):
        phot = phot_pipeline.phot()
    
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
