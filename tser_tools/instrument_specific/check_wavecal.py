# coding: utf-8
import instrument_specific
import matplotlib.pyplot as plt
import spec_pipeline
import pynrc
from astropy.io import fits, ascii
import numpy as np

def check_wavecal():
    sp = pynrc.nrc_utils.stellar_spectrum('A0V')

    spec = spec_pipeline.spec('parameters/spec_params/jwst/long_dark_otis_w_grism_source/long_dark_otis_grism_pairwise_sub_red_grism_03_smaller_ap.yaml')
    x,y,yerr = spec.get_avg_spec()
    head = fits.Header()
    head['FILTER'] = 'F322W2'
    wave = spec.wavecal(x,waveCalMethod='NIRCamTS',head=head,tserSim=True)

    pts = (sp.wave > np.min(wave * 1e4)) & (sp.wave < np.max(wave * 1e4))

    modelWave = sp.wave[pts] / 1e4

    plt.plot(modelWave,sp.flux[pts] * (modelWave/2.5)**3 * 0.1)
    plt.plot(wave,y)
    plt.show()
