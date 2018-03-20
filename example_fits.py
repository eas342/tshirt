import fit_tser_emcee
from astropy.io import ascii

def prepEclipse(withSys=False):
    """ Prepares Emcee run for fitting eclipse
    
    Parameters
    ---------------
    withSys: bool
        Add in a systematic to the time series?
    
    """
    
    dat = ascii.read('tser_data/phot/example_hd189733_f444w.csv')
    x = dat['Time (days)']
    if withSys == True:
        y = dat['Flux w/ Extra (ppm)'] * 1e-6
    else:
        y = dat['Flux (ppm)'] * 1e-6
    
    yerr = dat['Error (ppm)'] * 1e-6
    
    model = fit_tser_emcee.windModel()
    
    guess = [0.4,1100,800]
    spread = [0.3,300,300]
    
    mcObj = fit_tser_emcee.oEmcee(model,x,y,yerr,guess,spread,xLabel='Time (d)',
                                  yLabel='Flux (ppm)')
    return mcObj