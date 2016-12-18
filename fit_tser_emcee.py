import numpy as np
import emcee
import pdb
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import corner

class tdiffModel:
    """ A Two temperature surface model for fitting amplitude spectra of Brown Dwarfs
    A_lambda = (a_1 - a_2) (B(T1)/B(T2) - 1)/
                (a_1 + a_2) * B(T1)/B(T2) + (2 - a1 - a2)
    """
    def __init__(self,priorArray=None):
        self.name = "Two Temp Surface Flux Amplitude flambda"
        self.pnames = ['$\alpha$_1','$\alpha_2$','T$_1$','T$_2$']
        # self.formula = ("(a_1 - a_2) (B(T1)/B(T2) - 1)/"+
        #                 "(a_1 + a_2) B(T1)/B(T2)+ (2 - a1 - a2)")
        self.ndim = len(self.pnames)
    
    def planckNoConst(self,wavel,t):
        """ Planck function with no leading constants 
        
        Parameters
        ----------
        lambda: arr
            wavelength in microns
        t: arr
            temperature in Kelvin
        """
        return 1./(wavel**5 * (np.exp(14387.8/(wavel * t)) - 1.))
    
    def planckRatio(self,wavel,t1,t2):
        """ Finds the flux ratio of two Planck Functions"""
        ratio = self.planckNoConst(wavel,t1) / self.planckNoConst(wavel,t2)
        return ratio
    
    def evaluate(self,x,p):
        """ Evaluates the 2 temp model flambda """
        rat = self.planckRatio(x,p[2],p[3])
        numerator = (p[0] - p[1]) * (rat - 1.)
        denominator = (p[0] + p[1]) * rat + 2 - p[0] - p[1]
        return numerator/denominator
        
    def lnprior(self,p):
        """ Prior likelihood function """
        aCheck = (p[0:2] < 1) & (p[0:2] > 0)
        Tcheck = (p[3:5] > 1200) & (p[3:5] < 2500)
        if np.all(aCheck) & np.all(Tcheck):
            return 0
        else:
            return -np.inf

class sinModel:
    """ Simple sinusoidal model for fitting light curve.
        y(t) = A1 cos(2pi(t-t1)/tau) + Bt + C'
        p[0] = A, p[1] = t0, p[2]=tau, p[3]=B, p[4]=C
    """
    def __init__(self,priorArray=None):
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

class fSeries:
    """ A Fourier series model for fitting rotational modulations 
    
    y(t) = A1 cos(2pi(t-t1)/tau) + A2 cos(2pi(t-t2)/0.5tau) +... + Bt + C'
    Indices 0-2 are the time scale and baseline parameters
    
    Attributes
    ----------
    name: str
        Name of the model
    pnames: arr, str
        names of each parameter (LaTeX coding included)
        These begin with tau, B, C and then all A_i, then all t_i
    nbaseterms: int
        Number of timeline +baseline terms
    order: the number of terms in the Fourier series
    Aind: arr
        The indices for the amplitude terms
    tind: arr
        The indices for the 
    """
    def __init__(self,priorArray=None,order=2):
        self.name = "Fourier Series"
        pnames = [r'$\tau$','B','C']
        self.nbaseterms = len(pnames)
        self.order = order
        for i in range(order):
            pnames.append(r'A$_'+str(i+1)+r'$')
        self.Aind = np.arange(self.order) + self.nbaseterms
        for i in range(order):
            pnames.append(r't$_'+str(i+1)+r'$')
        self.tind = np.arange(self.order) + self.order + self.nbaseterms
        self.pnames = pnames
        self.formula = 'A1 cos(2pi(t-t1)/tau) + A2 cos(2pi(t-t2)/0.5tau) + ..+ Bt + C'
        self.ndim = len(self.pnames)
    
    def evaluate(self,x,p):
        """ Evaluates the cosine function"""
        freqArr = np.arange(self.order)+1.
        cosTerms = 0
        for Amp,freq,t in zip(p[self.Aind],freqArr,p[self.tind]):
            cosTerms += Amp/100. * np.cos(freq * 2. * np.pi * (x - t)/p[0])
        baseLine = p[1] * x + p[2]
        
        return cosTerms + baseLine
    
    def lnprior(self,p):
        """ Prior likelihood function"""
        ## Ensure positive amplitudes
        aCheck = p[self.Aind] > 0
        ## Ensure the offset is less than observation window
        tCheck = (p[self.tind] > -6.) & (p[self.tind] < 6.)
        ## Avoid harmonics
        tauCheck = (p[0] > 0. and p[0] < 5.)
        if np.all(aCheck) & np.all(tCheck) & np.all(tauCheck):
            return 0
        else:
            return -np.inf

def paramTable(model,values=None):
    """ Makes an astropy table of parameters, which is easy to read
    
    Parameters
    ----------
    values: arr
        Values to display next to parameter names
    """
    t = Table()
    t['Parameter'] = model.pnames
    t['Values'] = values
    return t

def lnprob(params, x, y, yerr, model):
    """ Likelihood function. A simple Chi-squared"""
    lp = model.lnprior(params)
    if np.isfinite(lp):
        ymodel = model.evaluate(x,params)
        return -0.5 * np.sum( ((y - ymodel)/yerr)**2)
    else:
        return -np.inf

class oEmcee():
    """ An MCMC object class to initialize and run an MCMC model
    
    Attributes
    ----------
    model: obj
        Model for the fitting
    x: arr
        X values
    y: arr
        Y values
    yerr: arr
        Y errors
    guess: arr
        Guess for the initial values of the model
    guessSig: arr
        Initial Gaussian sigmas to start walkers around the guess
    nwalkers: int, optional
        Number of walkers in MCMC fit
    p0: arr
        the initial arrays for the walkers
        sampler: the `emcee` Sampler object
    hasRun: bool
        whether or not the MCMC has been run
    xLabel: 
        Label for X axis
    yLabel:
        Label for Y axis
    title:
        Label for plot title
    
    """
    
    def __init__(self,model,x,y,yerr,guess,guessSig,nwalkers=50,xLabel='',yLabel='',
                 title=''):
        self.model = model
        self.x = x
        self.y = y
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.title = title
        self.yerr = yerr
        self.guess = np.array(guess)
        self.guessSig = np.array(guessSig)
        self.ndim = model.ndim
        self.nwalkers = nwalkers
        self.p0 = self.makeWalkers()
        self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,lnprob,
                                             args=[self.x,self.y,self.yerr,self.model])
        self.hasRun = False
    
    def makeWalkers(self):
        """ Makes the MCMC walkers """
        p0 = []
        for i in range(self.nwalkers):
            p0.append(self.guess + np.random.normal(0,1.0,self.ndim) * self.guessSig)
        return p0
    
    def showGuess(self,showResult=False,saveFile=None):
        """ Shows the guess against the input """
        plt.close('all')
        fig, ax = plt.subplots()
        ax.errorbar(self.x,self.y,yerr=self.yerr,fmt='o')
        if showResult == True:
            modelParam = self.results['Median']
        else:
            modelParam = self.guess
        ax.plot(self.x,self.model.evaluate(self.x,modelParam),linewidth=3.)
        ax.set_xlabel(self.xLabel)
        ax.set_ylabel(self.yLabel)
        ax.set_title(self.title)
        if saveFile == None:
            fig.show()
        else:
            fig.savefig(saveFile)
    
    def runMCMC(self,nBurn=500,nRun=1500):
        """ Runs the MCMC
        Example::
        mcObj = oEmcee(model,x,y,yerr,guess,guessSig)
        mcObj.runMCMC()
        mcObj.plotMCMC()
        
        """
        if nBurn != 0:
            self.pos, prob, state = self.sampler.run_mcmc(self.p0,nBurn)
            self.sampler.reset()
        else:
            self.sampler.reset()
        self.pos, prob, state = self.sampler.run_mcmc(self.pos,nRun)
        self.getResults()
        self.hasRun= True
    
    def getResults(self,lowPercent=15.9,highPercent=84.1):
        """ Get results from the chains in the sampler object """
        lower,medianV,upper = [], [], []
        for i in range(self.ndim):
            flatChain = self.sampler.flatchain[:,i]
            lower.append(np.percentile(flatChain,lowPercent))
            medianV.append(np.median(flatChain))
            upper.append(np.percentile(flatChain,highPercent))
        
        self.limPercents = [lowPercent,highPercent]
        t = Table()
        t['Parameter'] = self.model.pnames
        t['Lower'] = lower
        t['Median'] = medianV
        t['Upper'] = upper
        self.results = t
    
    def showResult(self):
        """ Shows the best-fit model from the median parameter values """
        self.runCheck()
        self.showGuess(showResult=True,saveFile='plots/best_fit.pdf')
        print(self.chisquare(self.results['Median']))
    
    def chisquare(self,param):
        """ Calculates the chi-squared"""
        chisquare = -2. * lnprob(param,self.x,self.y,self.yerr,self.model)
        dof = self.x.shape[0] - self.ndim
        return chisquare,chisquare/dof
    
    def runCheck(self):
        """A check that the MCMC has been run. If not, it runs"""
        if self.hasRun == False:
            print("Note: MCMC hasn't run yet and is now executing")
            self.runMCMC()
    
    def doCorner(self):
        """ Runs a corner plot """
        self.runCheck()
        
        fig = corner.corner(self.sampler.flatchain,labels=self.model.pnames,auto_bars=True,
                            quantiles=[0.159,0.841],show_titles=True)
        fig.savefig('plots/corner.pdf')
    
    def plotMCMC(self):
        """ Plots the MCMC histograms """
        self.runCheck()
        
        plt.close('all')
        fig, axArr = plt.subplots(1,self.ndim,figsize=(17,5))
        for i, ax in enumerate(axArr):
            postData = self.sampler.flatchain[:,i]
            ax.hist(postData,100,histtype='step')
            ax.set_xlabel(self.model.pnames[i])
            lowLimit = np.percentile(postData,5)
            highLimit = np.percentile(postData,95)
            showLow = lowLimit - (highLimit - lowLimit)
            showHigh = highLimit + (highLimit - lowLimit)
            ax.set_xlim(showLow,showHigh)
            ax.get_yaxis().set_visible(False)
        
        fig.show()
        
    def plotHistory(self):
        """ Shows the chain history """
        self.runCheck()
        plt.close('all')
        fig, axArr = plt.subplots(self.ndim,figsize=(10,10))
        for i, ax in enumerate(axArr):
            for j in range(self.sampler.chain.shape[0]):
                chainDat = self.sampler.chain[j,:,i]
                ax.plot(chainDat)
                ax.set_ylabel(self.model.pnames[i])
        fig.show()

def prepEmcee(doSeries=False):
    """ Prepares Emcee for run 
    
    Example usage::
    mcObj = fit_tser_emcee.prepEmcee(doSeries=True)
    mcObj.showResults()
    """
    dat = ascii.read('tser_data/timeser_1.08um_.txt',
                     names=['t','fl','flerr','model','resid'])
    x = np.array(dat['t'])
    y = np.array(dat['fl'])
    yerr = np.array(dat['flerr']) #* 3.
    
    if doSeries == True:
        order = 2
        if order == 1:
            guess = [4.1,0.,0.995,1.5,1.38]
            spread = [0.01,0.004,0.05,0.2,0.2]
        
        else:
            guess = [4.1,0.,0.995,1.5,0.4,1.38,1.38]
            spread = [0.01,0.004,0.05,0.2,0.2,0.2,0.2]
        model = fSeries(order=order)
        
    else:
        model = sinModel()
        guess = [1.5,1.38,4.1,0.,0.995]
        spread = [0.2,0.3,0.01,0.004,0.05]
    
    mcObj = oEmcee(model,x,y,yerr,guess,spread,xLabel='Time (hr)',yLabel='Normalized Flux',
                   title='Time Series at 1.08 $\mu$m')
    
    return mcObj

def prepEmceeSpec():
    """ Prepares Emcee run for """
    model = tdiffModel()
    



