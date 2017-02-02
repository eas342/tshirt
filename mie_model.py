import numpy as np
import miescatter

def extinct(wavel,rad=1.0,n=complex(1.825,-1e-4)):
    """
    Calculates the Mie extinction cross section Q_ext as a function of wavelength
    
    Arguments
    ------------------
    rad: float
        The radius of the particle
    wavel: float,arr
        The array of wavelengths to evaluate
    """
    sz = 2. * np.pi * rad/wavel
    qext,qsca,qabs,qbk,qpr,alb,g = miescatter.calcscatt(n,sz,n=len(sz))
    return qext
