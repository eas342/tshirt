from astropy.io import fits, ascii
from astropy.time import Time
import numpy as np
import astropy.units as u

def do_sim():
    np.random.seed(0)
    slope_file = 'noiseless/slope_NRCALONG_SUBGRISM64_F322W2_sim.fits'
    dat = fits.getdata(slope_file)
    orig_head = fits.getheader(slope_file)
    time = 10.
    sep = 20.
    read_noise = 13.
    signal = dat * time
    back = 2. * time ## a little higher
    t_start = Time('2020-04-13T20:00')
    
    sigma = np.sqrt(read_noise**2 + signal + back)
    
    for oneInt in np.arange(30):
        noise = sigma * np.random.randn(dat.shape[0],dat.shape[1])
        outdat = signal + back + noise
        
        t_obs = (t_start + oneInt * (time + sep) * u.s).fits
        
        outHDU = fits.PrimaryHDU(outdat)
        outHDU.header['BUNIT'] = ('e-', 'output unit (e-)')
        outHDU.header['READN'] = (read_noise, 'read noise (e-)')
        outHDU.header['FILTER'] = (orig_head['FILTER'], 'NIRCam filter')
        outHDU.header['DATE-OBS'] = (t_obs, 'obs start')
        outHDU.writeto('sim_data/ideal_noise_int_{:03d}.fits'.format(oneInt),overwrite=True)

if __name__ == "__main__":
    do_sim()
