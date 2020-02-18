import numpy as np

def ts_wavecal(pixels,tserSim=False,
               obsFilter='F444W',subarray='SUBGRISM64',grism='GRISM0'):
    """
    Simple analytic wavelength calibration for NIRCam grism time series
    """
    disp = -0.001 ## microns per pixel (toward positive X in raw detector pixels)
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
        undevPx = undevPx - 3. 

    
    if grism != 'GRISM0':
        raise Exception("Grism {} not available".format(grism))
    
    wavelengths = (pixels - undevPx) * disp + undevWav
    
    return wavelengths
    