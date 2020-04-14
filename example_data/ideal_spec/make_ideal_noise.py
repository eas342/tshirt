from astropy.io import fits, ascii
import numpy as np

def do_sim():
    np.random.seed(0)
    slope_file = 'noiseless/slope_NRCALONG_SUBGRISM64_F322W2_sim.fits'
    dat = fits.getdata(slope_file)
    orig_head = fits.getheader(slope_file)
    time = 10.
    read_noise = 13.
    signal = dat * time
    back = 2. * time ## a little higher
    
    sigma = np.sqrt(read_noise**2 + signal + back)
    
    for oneInt in np.arange(30):
        noise = sigma * np.random.randn(dat.shape[0],dat.shape[1])
        outdat = signal + back + noise
    
        outHDU = fits.PrimaryHDU(outdat)
        outHDU.header['BUNIT'] = ('e-', 'output unit (e-)')
        outHDU.header['READN'] = (read_noise, 'read noise (e-)')
        outHDU.header['FILTER'] = (orig_head['FILTER'], 'NIRCam filter')
        outHDU.writeto('sim_data/ideal_noise_int_{:03d}.fits'.format(oneInt),overwrite=True)

if __name__ == "__main__":
    do_sim()
