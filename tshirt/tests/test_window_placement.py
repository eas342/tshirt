import unittest
from tshirt.pipeline import phot_pipeline
import numpy as np

class borderChecks(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros([4,5])
    
    def test_inside(self):
        xres, yres = phot_pipeline.ensure_coordinates_are_within_bounds([2],[2.5],self.image)
        self.assertEquals(xres[0],2)
        self.assertEquals(yres[0],2.5)
    
    def test_outside(self):
        xres, yres = phot_pipeline.ensure_coordinates_are_within_bounds([-1.5],[2.5],self.image)
        self.assertEquals(xres[0],0)
        self.assertEquals(yres[0],2.5)

    def test_array(self):
        xres, yres = phot_pipeline.ensure_coordinates_are_within_bounds([-5,-2,3,4,7],
                                                                        [-7,0,1,3,12],self.image)
        self.assertTrue(np.allclose(xres,[0,0,3,4,4]))
        self.assertTrue(np.allclose(yres,[0,0,1,3,3]))
    
if __name__ == '__main__':
    unittest.main()
