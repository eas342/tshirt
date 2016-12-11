import numpy as np
import emcee
import pdb
from astropy.io import ascii
import matplotlib.pyplot as plt

class sinModel:
    """ Simple sinusoidal model for fitting light curve.
        p[0] = A, p[1] = t0, p[2]=tau, p[3]=B, p[4]=C
    """
    def __init__(self):
        self.name = "Sinusoidal with Baseline"
        self.pnames = ['A', 't$_0$', r'$\tau$', 'B', 'C']
        self.formula = 'A cos(2pi(t-t0)/tau) + Bt + c'
        self.ndim = len(self.pnames)
    
    def evaluate(self,x,p):
        """ Evaluates the cosine function"""
        cosTerm = p[0] * np.cos(2. * np.pi * (x - p[1])/p[2])
        baseLine = p[3] * x + p[4]
        return cosTerm + baseLine
    

def lnprob(params, x, y, yerr, model):
    """ Likelihood function. A simple Chi-squared"""
    ymodel = model.evaluate(x,params)
    return -0.5 * np.sum( ((y - ymodel)/yerr)**2)

def prepEmcee():
    """ Prepares Emcee for run """
    dat = ascii.read('tser_data/timeser_1.08um_.txt',
                     names=['t','fl','flerr','model','resid'])
    x = np.array(dat['t'])
    y = np.array(dat['fl'])
    yerr = np.array(dat['flerr'])
    model = sinModel()
    ndim = model.ndim
    nwalkers = 50
    guess = np.array([0.01,0.,3.,0.,1.0])
    
    p0 = []
    for i in range(nwalkers):
        p0.append(guess + np.random.normal(1,1.,ndim))
    
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x, y,yerr,model])
    
    return p0, sampler, model

def runMCMC():
    """ Runs the MCMC """
    p0, sampler, model = prepEmcee()
    sampler.run_mcmc(p0, 1000)
    pos, prob, state = sampler.run_mcmc(p0,1000)
    return sampler, model
    
def plotMCMC(sampler, model):
    """ Plots the MCMC histograms """
    fig, axArr = plt.subplots(1,model.ndim,figsize=(17,5))
    for i, ax in enumerate(axArr):
        ax.hist(sampler.flatchain[:,i],100,histtype='step')
        ax.set_xlabel(model.pnames[i])
        ax.get_yaxis().set_visible(False)
    fig.show()
