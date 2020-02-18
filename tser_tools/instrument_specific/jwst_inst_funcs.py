import numpy as np

def ts_wavecal(pixels,
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
    
    if grism != 'GRISM0':
        raise Exception("Grism {} not available".format(grism))
    
    wavelengths = (pixels - undevPx) * disp + undevWav
    
    return wavelengths
    