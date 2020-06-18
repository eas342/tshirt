from . import phot_pipeline
from . import spec_pipeline
from .phot_pipeline import robust_poly
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.time import Time
import astropy.coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
import numpy as np
from scipy.stats import binned_statistic
import pdb
from scipy.interpolate import interp1d

# try:
#     from astropy.modeling.models import BlackBody
#     bbFunc = True
# except ImportError:
#     bbFunc = False
#     print("Blackbody function doesn't work")

from astropy.utils.iers import conf
conf.auto_max_age = None


def k2_22(date='jan25',showTiming=False,detrend=True,doBin=False):
    dat = ascii.read('tser_data/reference_dat/k2-22_phased_transit.txt',
                     names=['phase','flux','flux err','junk'],
                     format='fixed_width',delimiter=' ')

    if date == 'jan25':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_25_phot_lbc_proc.yaml')
        t0 = Time('2020-01-25T09:30:27.06')
    elif date == 'jan25-luci2':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_25_phot_luci2.yaml')
        t0 = Time('2020-01-25T09:30:27.06')
    elif date == 'jan28':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_28_phot_lbc.yaml')
        t0 = Time('2020-01-28T10:40:45.81')
    elif date == 'jan28-luci2':
        #phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_28_phot_luci2.yaml')
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_28_phot_luci2_modflat.yaml')
        #phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_01_28_phot_luci2_nanreplace.yaml') 
        t0 = Time('2020-01-28T10:40:45.81')
    elif date == 'feb20':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_02_20_phot_lbc.yaml')
        t0 = Time('2020-02-20T07:25:38.77')
    elif date == 'feb20-luci2':
        phot = phot_pipeline.phot('parameters/phot_params/lbt/k2-22_UT_2020_02_20_phot_luci2.yaml')
        t0 = Time('2020-02-20T07:25:38.77')
    else:
        raise Exception("Unrecognized date {}".format(date))
    
    if 'luci' in date:
        band = 'LUCI2 Ks-band'
        suffix = ''
    else:
        band = 'LBC g-band'
        suffix = '_lbc'
    
    HDUList = fits.open(phot.photFile)
    photHDU = HDUList['PHOTOMETRY']
    photArr = photHDU.data
    errArr = HDUList['PHOT ERR'].data
    head = photHDU.header
    
    jdHDU = HDUList['TIME']
    jdArr_ut = jdHDU.data
    
    
    coord = SkyCoord("11:17:55.88", "+02:37:08.6",
                            unit=(u.hourangle, u.deg), frame='icrs')
    loc = astropy.coordinates.EarthLocation.of_site('lbt') 
    times = Time(jdArr_ut, format='jd',
                 scale='utc', location=loc)  
    ltt_bary = times.light_travel_time(coord)  
    bjd = times.utc + ltt_bary
    jdArr = bjd.jd
    
    timeHead = jdHDU.header
    
    jdRef = phot.param['jdRef']
    
    
    fig, ax = plt.subplots(figsize=(4,3.5))
    
    yCorrected, yCorrected_err = phot.refSeries(photArr,errArr,excludeSrc=None)
    x = jdArr - jdRef
    
    if detrend == True:
        T0 = 2456811.1208
        P = 0.381078
        phase = np.mod((jdArr - T0)/P,1.0)
        fitp = (phase < -0.1) | (phase > 0.1)
        polyBase = robust_poly(x[fitp],yCorrected[fitp],2,sigreject=2)
        yBase = np.polyval(polyBase,x)
        
        yShow = yCorrected / yBase
    else:
        yShow = yCorrected
    
    ax.plot(x,yShow,'o',label=band)
    
    if doBin == True:
        goodP = np.isfinite(yShow)
        span = np.max(x) - np.min(x)
        nBin = np.int(span * 24. * 60./10.)
        yBins = Table()
        for oneStatistic in ['mean','std','count']:
            if oneStatistic == 'std':
                statUse = np.std
            else: statUse = oneStatistic
            
            yBin, xEdges, binNum = binned_statistic(x[goodP],yShow[goodP],
                                                    statistic=statUse,bins=nBin)
            yBins[oneStatistic] = yBin
    
        ## Standard error in the mean
        stdErrM = yBins['std'] / np.sqrt(yBins['count'])
    
        xbin = (xEdges[:-1] + xEdges[1:])/2.
        ax.errorbar(xbin,yBins['mean'],yerr=stdErrM,marker='s',markersize=3.,
                label='binned')
    #phot.plot_phot(refCorrect=True,yLim=[0.97,1.02],fig=fig,ax=ax)
    
    
    dat['time'] = (t0.jd - jdRef) + (dat['phase'] - 1.5) * 0.381078 
    
    pts = (dat['time'] < np.max(x) + 0.05) & (dat['time'] > np.min(x))
    
    ax.plot(dat['time'][pts],dat['flux'][pts],color='red',label='Kepler K2-22 Avg')
    ax.set_xlabel('Time (JD - {})'.format(jdRef))
    ax.set_ylabel('Normalized Flux')
    ax.set_ylim(0.991,1.005)
    
    if showTiming == True:
        cenTime = t0.jd - jdRef
        ax.axvline(cenTime, color='orange')
        
        ax.axvline(cenTime - 2.04 / 24.,color='green',label='Good Start/End')
        ax.axvline(cenTime - 1.00 / 24.,color='purple',label='Absolute Latest Start/End')
        ax.axvline(cenTime + 1.00 / 24.,color='purple')
        ax.axvline(cenTime + 2.04 / 24.,color='green')
        extraInfo = '_timing'
    else:
        extraInfo = ''
    ax.legend()
    
    fig.savefig('plots/photometry/custom_plots/k2_22_ut{}{}{}.pdf'.format(date,extraInfo,suffix),
                bbox_inches='tight')
    plt.close(fig)

#def fringing_function(x,amp=0.1,period=0.09,periodSlope=.04,offset=0.1):
def fringing_function(x,amp=0.1,period=6.0,periodSlope=2.0,offset=0.1,phase=0.0):
    
    L = 315. ##microns
    n_real = 2.7
    E_0 = np.sin(2. * np.pi * L * n_real / x)
    modelY = 1.0 - offset + amp * E_0**2
    return modelY

def show_fringing():
    modelX = np.array(np.arange(0,2048),dtype=np.float)
    for oneTest in ['otis','cv3']:
        if oneTest == 'otis':
            spec = spec_pipeline.spec('parameters/spec_params/jwst/otis_grism/otis_grism_ts_w_flats_PPP_w_f322w2.yaml')
            numSplineKnots = 200
            fringAmpGuess = 0.006
            fringOffset = 0.013
            waveOffset = 0
            yLim = [0.98,1.008]
        elif oneTest == 'cv3':
            spec = spec_pipeline.spec('parameters/spec_params/jwst/grism_cv3/f322w2_grism_example.yaml')
            numSplineKnots = 400
            fringAmpGuess = 0.1
            fringOffset = 0.2
            waveOffset = 0.0
            yLim = [0.8,1.2]
        else:
            raise Exception("Unrecognized test data {}.".format(oneTest))
        
        #modelY = fringing_function(modelX,amp=fringAmpGuess,offset=fringOffset)
        
        x, y, yerr = spec.get_avg_spec()
        lam = spec.wavecal(x) - waveOffset
        modelY = fringing_function(lam,amp=fringAmpGuess,offset=fringOffset)
        
        normY = spec.norm_spec(x,y,numSplineKnots=400)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.set_xlim(1000,1800)
        ax.plot(x,normY,label='Extracted Spectrum')
        ax.plot(modelX,modelY,label='Thin Film Model')
        reprPoint = 1500
        ax.errorbar([modelX[reprPoint]],[np.nanpercentile(normY,99)],yerr=yerr[reprPoint]/y[reprPoint],
                    color='green',fmt='o',markerfacecolor='none',label='Representative Error')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        ax.set_ylim(yLim[0],yLim[1])
        
        ## show the wavelengths
        wave_labels = np.arange(2.5,4.1,0.1)
        ax2 = ax.twiny()
        f = interp1d(lam,x)
        new_tick_locations = f(wave_labels)
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(wave_labels)
        ax2.set_xlabel(r"Wavelength ($\mu$m)")
        ax2.set_xlim(ax.get_xlim())
        
        fig.show()
        #pdb.set_trace()
        fig.savefig('plots/spectra/custom_plots/fringing_grism_{}.pdf'.format(oneTest),bbox_inches='tight')
        plt.close(fig)


def check_wavecal(otis=False):
    """
    Check the wavelength cal 
    """
    dat = ascii.read('spectra/specific/reference_data/F322W2_nrc_only_throughput_moda_sorted.txt')
    
    yFilt = dat['Throughput']
    yFiltNorm = yFilt / np.median(yFilt)
    
    plt.plot(dat['Wavelength_microns'],yFiltNorm)
    
    if otis == True:
        spec = spec_pipeline.spec('parameters/spec_params/jwst/otis_grism/otis_grism_ts_w_flats_PPP_w_f322w2.yaml')
        bb_temp = 300
        spec_offset = 0.0
    else:
        bb_temp = 500
        spec = spec_pipeline.spec('parameters/spec_params/jwst/grism_cv3/f322w2_grism_example.yaml')
        spec_offset = 0.0
    
    x, ySpec, yerr = spec.get_avg_spec()
    lam = spec.wavecal(x) - spec_offset
    
    exp_const = const.h * const.c / ( const.k_B * bb_temp * u.K)
    exp_const_mic = exp_const.to(u.micron).value
    bb_wien = np.exp(- exp_const_mic/lam) / lam**5
    #bb = BlackBody(temperature=330*u.K)
    #wav = lam * u.micron
    y = ySpec / bb_wien#(wav)
    
    yNorm = y / np.nanmedian(y)
    plt.plot(lam,yNorm)
    plt.show()
    
    