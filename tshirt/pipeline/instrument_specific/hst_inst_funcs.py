from __future__ import print_function, division
import numpy as np
from scipy.interpolate import interp1d

def hstwfc3_wavecal(dispIndices,xc0=None,yc0=None,DirIm_x_appeture=522,DirIm_y_appeture=522,SpecIm_x_appeture=410,
                    SpecIm_y_appeture=522,subarray=128):
    """
        Wavelength calibration to turn the dispersion pixels into wavelengths
        
        Parameters
        -------------
        dispIndices: numpy array 
            Dispersion Middle X-axis values
        xc0: float
            X coordinate of direct image centroid
        yc0: float
            Y coordinate of direct image centroid
        DirIm_x_appeture: float
            Chip Reference X-Pixel dependent upon direct image centroid apeture 
        DirIm_y_appeture: float
            Chip Reference Y-Pixel dependent upon direct image centroid apeture
        SpecIm_x_appeture: float
            Chip Reference X-Pixel dependent upon spectral image centroid apeture 
        SpecIm_y_appeture: float
            Chip Reference Y-Pixel dependent upon spectral image centroid apeture
        subarray:
            Length of detector array 
            
        Note that direct image and spectral image should be taken with the
        same aperture. If not, please adjust the centroid measurement
        according to table in: https://www.stsci.edu/hst/instrumentation/focus-and-pointing/fov-geometry
    """
    
    #accounting for the offset from the direct image centroid
    x_offset = (DirIm_x_appeture - SpecIm_x_appeture)
    y_offset = (DirIm_y_appeture - SpecIm_y_appeture)

    xc = xc0 - (x_offset)
    yc = yc0 - (y_offset)

    coord0 = (1014 - subarray) // 2
    xc = xc + coord0
    yc = yc + coord0

    DLDP0 = [8949.40742544, 0.08044032819916265]
    DLDP1 = [44.97227893276267,
             0.0004927891511929662,
             0.0035782416625653765,
             -9.175233345083485e-7,
             2.2355060371418054e-7,  
             -9.258690000316504e-7]

    # calculate field dependent dispersion coefficient
    p0 = DLDP0[0] + DLDP0[1] * xc
    p1 = DLDP1[0] + DLDP1[1] * xc + DLDP1[2] * yc + \
         DLDP1[3] * xc**2 + DLDP1[4] * xc * yc + DLDP1[5] * yc**2

    dx = np.arange(1014) - xc
    wavelength = (p0 + dx * p1)/10000 #in microns
    if subarray < 1014:
        i0 = (1014 - subarray) // 2
        wavelength = wavelength[i0: i0 + subarray]
        f = interp1d(np.arange(subarray),wavelength)
        wavelengths = f(dispIndices)
    return wavelengths