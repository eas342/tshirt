import phot_pipeline
import spec_pipeline
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.time import Time
import numpy as np
import pdb

def k2_22(date='jan25'):
    dat = ascii.read('tser_data/reference_dat/k2-22_phased_transit.txt',
                     names=['phase','flux','flux err','junk'],
                     format='fixed_width',delimiter=' ')

    if date == 'jan25':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_25_phot_lbc_proc.yaml')
        t0 = Time('2020-01-25T09:30:27.06')
    elif date == 'jan28':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_28_phot_lbc.yaml')
        t0 = Time('2020-01-28T10:40:45.81')
    else:
        raise Exception("Unrecognized date {}".format(date))
    
    HDUList = fits.open(phot.photFile)
    photHDU = HDUList['PHOTOMETRY']
    photArr = photHDU.data
    errArr = HDUList['PHOT ERR'].data
    head = photHDU.header
    
    jdHDU = HDUList['TIME']
    jdArr = jdHDU.data
    timeHead = jdHDU.header
    
    jdRef = phot.param['jdRef']
    
    
    fig, ax = plt.subplots()
    
    yCorrected, yCorrected_err = phot.refSeries(photArr,errArr,excludeSrc=None)
    x = jdArr - jdRef
    ax.plot(x,yCorrected,label='LBC g-band')
    
    #phot.plot_phot(refCorrect=True,yLim=[0.97,1.02],fig=fig,ax=ax)
    
    
    dat['time'] = (t0.jd - jdRef) + (dat['phase'] - 1.5) * 0.381078 
    
    pts = (dat['time'] < np.max(x) + 0.05) & (dat['time'] > np.min(x))
    
    ax.plot(dat['time'][pts],dat['flux'][pts],color='red',label='Kepler K2-22 Avg')
    ax.set_xlabel('Time (JD - {})'.format(jdRef))
    ax.set_ylabel('Normalized Flux')
    ax.legend()
    ax.set_ylim(0.991,1.005)
    
    fig.savefig('plots/photometry/custom_plots/k2_22_ut{}_lbc.pdf'.format(date))
    plt.close(fig)
    
def show_fringing():
    spec = spec_pipeline.spec('parameters/spec_params/jwst/otis_grism/otis_grism_ts_w_flats_PPP_w_f322w2.yaml')
    x, y, yerr = spec.get_avg_spec()
    normY = spec.norm_spec(x,y,numSplineKnots=200)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_xlim(1000,1800)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Normalized Flux')
    ax.plot(x,normY)
    fig.savefig('plots/spectra/custom_plots/otis_grism_fringing.pdf',bbox_inches='tight')
    