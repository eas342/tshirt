import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d


class kic1255Model:
    """ A Kepler LC model from the Kepler Short cadence data
    
    """
    def __init__(self):
        self.name = "Kepler LC Model from Kepler SC data"
        self.pnames = [r'A','B']
        self.formula = "(A * f(phase) + 1.0) * (1.0 * B)"
        
        self.kepFile = 'tser_data/reference_dat/avg_bin_kep.fits'
        self.kepDat = Table.read(self.kepFile)
        
        self.finterp = interp1d(self.kepDat['BINMID'],self.kepDat['YBIN'] - 1.0,
                                bounds_error=False,fill_value=0.0)
        self.ndim = len(self.pnames)
    
    def evaluate(self,x,inputP):
        """ Evaluates the SC model with an interpolating function 
        
        Parameters
        -----------
        x: np.array
            The input phase 
        p: np.array or list
            the parameter array
            p[0] = Amplitude
            p[1] = linear baseline
        """
        p = np.array(inputP)
        return (p[0] * self.finterp(x) + 1.0) * (1.0 + p[1])
    
    
    def lnprior(self,inputP):
        """ Prior likelihood """
        
        p = np.array(inputP)
        finiteCheck = np.isfinite(inputP)
        
        if np.all(finiteCheck):
            return 0
        else:
            return -np.inf