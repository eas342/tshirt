import numpy as np
import emcee
import pdb
from astropy.io import ascii
import matplotlib.pyplot as plt
import corner

class sinModel:
    """ Simple sinusoidal model for fitting light curve.
        p[0] = A, p[1] = t0, p[2]=tau, p[3]=B, p[4]=C
    """
    def __init__(self):
        self.name = "Sinusoidal with Baseline"
        self.pnames = ['A$_1$', 't$_1$', r'$\tau$', 'B', 'C']
        self.formula = 'A1 cos(2pi(t-t1)/tau) + Bt + C'
        #self.pnames = ['A$_1$', 't$_1$', r'$\tau$', 'B', 'C','A$_2$','t$_2$']
        #self.formula = 'A1 cos(2pi(t-t1)/tau) + A2 cos(2pi(t-t2)/0.5tau) + Bt + C'
        self.ndim = len(self.pnames)
    
    def evaluate(self,x,p):
        """ Evaluates the cosine function"""
        cosTerm = (p[0]/100. * np.cos(2. * np.pi * (x - p[1])/p[2]))
        #cosTerm2 = (p[5]/100. * np.cos(2. * np.pi * (x - p[6])/(p[2] * 0.5)))
        baseLine = p[3] * x + p[4]
        return cosTerm + baseLine #+cosTerm2
    
    def lnprior(self,p):
        """ Prior likelihood function"""
        pCheck = []
        pCheck.append(p[0] > 0)
        ## Ensure the offset is less than observation window
        pCheck.append(p[1] > -6. and p[1] < 6.) 
        ## Avoid harmonics
        pCheck.append(p[2] > 0. and p[2] < 5.)
        #pCheck.append(p[5] > 0)
        #pCheck.append(np.abs(p[6]) < 7.)
        if np.sum(pCheck) == len(pCheck):
            return 0
        else:
            return -np.inf

def lnprob(params, x, y, yerr, model):
    """ Likelihood function. A simple Chi-squared"""
    lp = model.lnprior(params)
    if np.isfinite(lp):
        ymodel = model.evaluate(x,params)
        return -0.5 * np.sum( ((y - ymodel)/yerr)**2)
    else:
        return -np.inf

def prepEmcee(showGuess=False):
    """ Prepares Emcee for run """
    dat = ascii.read('tser_data/timeser_1.08um_.txt',
                     names=['t','fl','flerr','model','resid'])
    x = np.array(dat['t'])
    y = np.array(dat['fl'])
    yerr = np.array(dat['flerr']) #* 3.
    model = sinModel()
    ndim = model.ndim
    nwalkers = 50
    
    #guess = np.array([0.6,2.1,1.9,0,1])
    guess = np.array([1.5,1.38,4.1,0.,0.995])#0.7,1.0])
    #guess = np.array([1.5,1.38,4.1,0.,0.995])
    spread = np.array([0.2,0.3,0.01,0.004,0.05])
    
    
    if showGuess == True:
        plt.close('all')
        fig, ax = plt.subplots()
        ax.errorbar(x,y,yerr=yerr,fmt='o')
        ax.plot(x,model.evaluate(x,guess))
        fig.show()
    
    p0 = []
    for i in range(nwalkers):
        p0.append(guess + np.random.normal(0,1.0,ndim) * spread)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x, y,yerr,model])
    
    return p0, sampler, model

def getMeds(sampler):
    """ Get medians """
    ndim = sampler.flatchain.shape[1]
    result = []
    for i in range(ndim):
        result.append(np.median(sampler.flatchain[:,i]))
    return result
    
def runMCMC():
    """ Runs the MCMC
    Example::
    sampler, model = fit_tser_emcee.runMCMC()
    fit_tser_emcee.plotMCMC(sampler,model)
    
     """
    p0, sampler, model = prepEmcee()
    ## Burn-in for 500 steps, then discard burn-in step
    pos, prob, state = sampler.run_mcmc(p0,500)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos,1500)
    return sampler, model
    
def plotHistory(sampler, model):
    """ Shows the chain history """
    plt.close('all')
    fig, axArr = plt.subplots(model.ndim,figsize=(10,10))
    for i, ax in enumerate(axArr):
        for j in range(sampler.chain.shape[0]):
            chainDat = sampler.chain[j,:,i]
            ax.plot(chainDat)
            ax.set_ylabel(model.pnames[i])
    fig.show()

def doCorner(sampler,model):
    """ Runs a corner plot """
    fig = corner.corner(sampler.flatchain,labels=model.pnames,auto_bars=False,
                        quantiles=[0.159,0.841],show_titles=True)
    fig.savefig('plots/corner.pdf')

def plotMCMC(sampler, model):
    """ Plots the MCMC histograms or history """
    plt.close('all')
    fig, axArr = plt.subplots(1,model.ndim,figsize=(17,5))
    for i, ax in enumerate(axArr):
        postData = sampler.flatchain[:,i]
        ax.hist(postData,100,histtype='step')
        ax.set_xlabel(model.pnames[i])
        lowLimit = np.percentile(postData,5)
        highLimit = np.percentile(postData,95)
        showLow = lowLimit - (highLimit - lowLimit)
        showHigh = highLimit + (highLimit - lowLimit)
        ax.set_xlim(showLow,showHigh)
        ax.get_yaxis().set_visible(False)
        
    fig.show()
