import numpy as np
import pdb

def ts_wavecal(pixels,tserSim=False,
               obsFilter='F444W',subarray='SUBGRISM64',grism='GRISM0'):
    """
    Simple analytic wavelength calibration for NIRCam grism time series
    """
    disp = -0.0010035 ## microns per pixel (toward positive X in raw detector pixels, used in pynrc)
    undevWav = 4.0 ## undeviated wavelength
    
    if obsFilter == 'F444W':
        undevPx = 1096
    elif obsFilter == 'F322W2':
        undevPx = 467
    else:
        raise Exception("Filter {} not available".format(obsFilter))

    if tserSim == True:
        ## fudge factor for wavelength calibration in time series simulation at the shorter wavelengths
        ## the simulation is slightly off from expectations. In reality we'll want to use wavecal source anyway
        undevPx = undevPx + 2.
    

    
    if grism != 'GRISM0':
        raise Exception("Grism {} not available".format(grism))
    
    wavelengths = (pixels - undevPx) * disp + undevWav
    
    return wavelengths

def ts_wavecal_quick_nonlin(pixels,obsFilter='F322W2'):
    """
    Simple inefficient polynomial
    """
    
    if obsFilter == 'F322W2':
        x = pixels
        wavelengths = 2.39610143 + 9.5273e-4 * x + 1.5e-8 * x**2 + -2.89e-12 * x**3
    else:
        raise Exception("Filter {} not available".format(obsFilter))
    
    return wavelengths
    

def ts_grismc_sim(pixels):
    """
    Simple analytic wavelength calibration for Simulated GRISMC data
    """
    disp = 0.0010035 ## microns per pixel (toward positive X in raw detector pixels, used in pynrc)
    undevWav = 4.0 ## undeviated wavelength
    undevPx = 1638.33
    
    wavelengths = (pixels - undevPx) * disp + undevWav
    
    return wavelengths
    