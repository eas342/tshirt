import numpy as np
import emcee
import pdb
import os
import glob
from astropy.io import ascii
from astropy.table import Table
from matplotlib import style as mplStyle
mplStyle.use('classic')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.collections import LineCollection
from matplotlib import gridspec
import corner
import yaml
import re
import string
import mie_model
import pickle
from copy import deepcopy
import warnings
try:
    import spiderman
except ImportError:
    warnings.warn('Could not load spiderman models.')
import phot_pipeline
import fit_models

def sanitize_param(inputP):
    """ Sanitizes the input parameters"""
    return np.array(inputP)
    
class tdiffModel:
    """ A Two temperature surface model for fitting amplitude spectra of Brown Dwarfs
    A_lambda = (beta) (B(T1)/B(T2) - 1)/
                (2 a_1 - beta) * B(T1)/B(T2) + (2 - 2 a1 + beta)
    """
    def __init__(self,priorArray=None):
        self.name = "Two Temp Surface Flux Amplitude flambda"
        self.pnames = [r'$\alpha_1$',r'$\beta$','T$_1$','T$_2$']
        self.formula = ("(beta) (B(T1)/B(T2) - 1)/"+
                         "(2 a_1 - beta) B(T1)/B(T2)+ (2 - 2a1 + beta)")
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
        """ Evaluates the 2 temp model flambda. Puts it in percent """
        rat = self.planckRatio(x,p[2],p[3])
        numerator = p[1] * (rat - 1.)
        denominator = (2 * p[0] - p[1]) * rat + 2. - 2. * p[0] + p[1]
        return numerator/denominator * 100.
        
    def lnprior(self,inputP):
        """ Prior likelihood function """
        p = sanitize_param(inputP)
        aCheck = (p[0:2] < 1) & (p[0:2] > 0)
        #Tcheck = (p[3:5] > 1200) & (p[3:5] < 2500)
        Tcheck = (p[3:5] > 100) & (p[3:5] < 1e4)
        if np.all(aCheck) & np.all(Tcheck):
            return 0
        else:
            return -np.inf

class mieModel:
    """ A Mie extinction model for the brown dwarf rotation curves
    composition = Simple - use a Enstatite single complex index of refraction
    logNorm - whether to use a log-normal distribution of particles
    maxRad - a prior for the maximum particle radius
    variableSigma: bool
        Whether to vary the log-normal sigma parameter
    """
    def __init__(self,composition='simple',logNorm=True,maxRad = 5.,
                 variableSigma=False):
        self.name = "Mie extinction model for flux amplitude"
        self.pnames = [r'B','r']
        self.formula = "B * Qext(wavel,tau)"
        
        if (logNorm == True) & (variableSigma == True):
            ## Add a sigma free parameter
            ## Only works for Log-normal particle distribution
            self.pnames.append('$\sigma$')
            self.formula = "B * Qext(wavel,tau,sigma)"
            self.variableSigma = True
        else:
            self.variableSigma = False
            
        self.logNorm = logNorm
        self.ndim = len(self.pnames)
        self.maxRad = maxRad
    
    def evaluate(self,x,p):
        """ Evaluates the Mie extinction model assuming 
        that the wavelength and radius are the same parameters
        see obj.pnames for parameter names"""
        if (self.variableSigma == True) & (self.logNorm == True):
            Qext = mie_model.extinct(x,p[1],logNorm=self.logNorm,s=p[2])
        elif self.logNorm == True:
            Qext = mie_model.polyExtinct(x,rad=p[1])
        else:
            Qext = mie_model.extinct(x,p[1],logNorm=self.logNorm)
        
        return p[0] * Qext
    
    def lnprior(self,inputP):
        p = sanitize_param(inputP)
        zeroCheck = inputP > 0
        maxRadCheck = p[1] < self.maxRad
        if self.variableSigma == True:
            sigCheck = (p[2] > 0.) & (p[2] < 2.5)
        else:
            sigCheck = True
        
        if np.all(zeroCheck & maxRadCheck & sigCheck):
            return 0
        else:
            return -np.inf

class windModel:
    """ A simple wrapper about the Spiderman Phase-curve model
    For now, it's hard-coded for HD 189733
    """
    def __init__(self):
        spider_params = spiderman.ModelParams(brightness_model="zhang")
        spider_params.n_layers= 8
        spider_params.t0= -2.21857567/2.               # Central time of PRIMARY transit [days]
        spider_params.per= 2.21857567       # Period [days]
        spider_params.a_abs= 0.03099        # The absolute value of the semi-major axis [AU]
        spider_params.inc= 85.58            # Inclination [degrees]
        spider_params.ecc= 0.0              # Eccentricity
        spider_params.w= 90                 # Argument of periastron
        spider_params.rp= 0.15463           # Planet to star radius ratio
        spider_params.a= 8.81               # Semi-major axis scaled by stellar radius
        spider_params.p_u1= 0               # Planetary limb darkening parameter
        spider_params.p_u2= 0               # Planetary limb darkening parameter
        spider_params.xi= 0.3       # Ratio of radiative to advective timescale
        spider_params.T_n= 900      # Temperature of nightside
        spider_params.delta_T= 500  # Day-night temperature contrast
        spider_params.T_s = 5040    # Temperature of the star
        spider_params.l1 = 4.0e-6       # The starting wavelength in meters
        spider_params.l2 = 5.0e-6       # The ending wavelength in meters
        
        self.spider_params = spider_params
        self.name = 'Spiderman Wind Model'
        self.pnames = [r'$\xi$',r'T$_n$',r'$\Delta$T']
        self.formula = "Zhang & Showman Analytic"
        self.ndim = len(self.pnames)
        
    def evaluate(self,x,p):
        """ Evaluates the Zhang & Showman simplified phase curve"""
        self.spider_params.xi = p[0]
        self.spider_params.T_n = p[1]
        self.spider_params.delta_T = p[2]
        return self.spider_params.lightcurve(x)
    
    def lnprior(self,inputP):
        p = sanitize_param(inputP)
        pCheck = []
        pCheck.append(p[0] > 0)
        pCheck.append(p[1] > 0)
        if np.sum(pCheck) == len(pCheck):
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
    
    def lnprior(self,inputP):
        """ Prior likelihood function"""
        p = sanitize_param(inputP)
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
        The indices for the time offset of maximum deviation for each term
    """
    def __init__(self,srcData=None,order=2,useAirmass=False):
        self.name = "Fourier Series"
        pnames = [r'$\tau$','B','C']
        
        self.useAirmass = useAirmass
        if useAirmass == True:
            pnames.append('D')
        
        self.nbaseterms = len(pnames)
        self.order = order
        for i in range(order):
            pnames.append(r'A$_'+str(i+1)+r'$')
        self.Aind = np.arange(self.order) + self.nbaseterms
        for i in range(order):
            pnames.append(r't$_'+str(i+1)+r'$')
        self.tind = np.arange(self.order) + self.order + self.nbaseterms
        self.pnames = pnames
        
        fSeriesPart = '(1 + A1 cos(2pi(t-t1)/tau) + A2 cos(2pi(t-t2)/0.5tau) + ..+) '
        if useAirmass == True:
            baselinePart = '* (Bt + C + D a(t)'
        else:
            baselinePart = '* (Bt + C)'
        
        self.formula = fSeriesPart + baselinePart
        
        self.ndim = len(self.pnames)
        if srcData == None:
            self.priorSet = False
        else:
            self.get_prior_params(srcData)
        
    def get_prior_params(self,srcData):
        """ Reads in the YAML file for parameters and sets them """
        with open('parameters/fit_params.yaml') as paramFile:
            priorP = yaml.load(paramFile)
            self.priorSet = True
            self.priorPeriod = priorP[srcData]['prior']['periodRange']
    
    def evaluate(self,x,p):
        """ Evaluates the cosine function"""
        freqArr = np.arange(self.order)+1.
        cosTerms = 0
        
        if self.useAirmass == True:
            xTime = x[0,:]
            xAirmass = x[1,:]
        else:
            xTime = x
        
        for Amp,freq,t in zip(p[self.Aind],freqArr,p[self.tind]):
            cosTerms += Amp/100. * np.cos(freq * 2. * np.pi * (xTime - t)/p[0])
        
        baseLine = p[1] * xTime + p[2]
        if self.useAirmass == True:
            baseLine = baseLine + p[3] * x[1,:]
        
        return (cosTerms + 1.) * baseLine
    
    def lnprior(self,inputP):
        """ Prior likelihood function"""
        ## Ensure positive amplitudes
        p = sanitize_param(inputP)
        #aCheck = p[self.Aind] > 0
        aCheck = True
        ## Ensure the offset is less than half the observation window to avoid
        ## negative amplitude and opposite phase
        offRange = 0.5 * p[0]#4.2
        ## For higher order terms, the time offsets must have correspondingly small windows
        windowRange = offRange/np.arange(1,self.order+1,dtype=float)
        tCheck = ((p[self.tind] > -1. * windowRange) & 
                  (p[self.tind] < 1. * windowRange))
        ## Avoid harmonics
        if self.priorSet == True:
            tauCheck = (p[0] > self.priorPeriod[0] and p[0] < self.priorPeriod[1])
        else:
            tauCheck = True
        
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
            Can be multi-dimensional
    xplot: arr
        X values to plot - one row of a possible multi-dimensional x
        If x is a 1D array, xplot is the same as x
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
                 title='',plotIndx=0):
        self.model = model
        self.x = x
        
        ## IF x is multi-dimensional, choose one row to plot
        self.plotIndx = plotIndx
        if x.ndim > 1:
            self.xplot = self.x[plotIndx,:]
        else:
            self.xplot = self.x
        
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
    
    def showGuess(self,showParamset=None,saveFile=None,ax=None,fig=None,
                  residual=False,figsize=None):
        """ Shows the guess or specified parameters against the input
        
        Parameters
        ---------------
        showParamset: arr
            the parameters to show a model for. If `None`, it shows the guess
        saveFile: str
            where to save the plot. If `None`, it doesn't save a file
        ax: obj
            matplotlib axis object of plot to use
        fig: obj
            matplotlib figure object of plot to use
        residual: bool
            whether to show the residuals
                  """
        plt.close('all')
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if showParamset is None:
            modelParam = self.guess
        else:
            modelParam = showParamset
        
        xmodel = np.linspace(np.min(self.xplot),np.max(self.xplot),1024)
        if self.x.ndim > 1:
            ## Interpolate other dimensions for just as many points
            xEntry = np.zeros([self.x.shape[0],xmodel.size])
            sortedInd = np.argsort(self.x[self.plotIndx,:])
            for oneRow in range(self.x.shape[0]):
                xInterp = np.interp(xmodel,self.x[self.plotIndx,sortedInd],self.x[oneRow,sortedInd])
                xEntry[oneRow,:] = xInterp
        else:
            xEntry = xmodel
        
        if residual == True:
            yModel = self.model.evaluate(self.x,modelParam)
            yShow = self.y - yModel
            print('Median Error = '+str(np.median(self.yerr)))
            print('Standard Dev Residuals = '+str(np.nanstd(yShow)))
            ax.errorbar(self.xplot,yShow,yerr=self.yerr,fmt='o')
        else:
            ax.errorbar(self.xplot,self.y,yerr=self.yerr,fmt='o')
            yModel = self.model.evaluate(xEntry,modelParam)
            yShow = yModel
            
            ax.plot(xmodel,yShow,linewidth=3.)
        ax.set_xlabel(self.xLabel)
        ax.set_ylabel(self.yLabel)
        ax.set_title(self.title)
        if saveFile is None:
            fig.show()
        else:
            fig.savefig(saveFile,bbox_inches='tight')
    
    def runMCMC(self,nBurn=500,nRun=1500):
        """ Runs the MCMC
        Example::
        mcObj = oEmcee(model,x,y,yerr,guess,guessSig)
        mcObj.runMCMC()
        mcObj.plotMCMC()
        
        If you look at the chains and they're not converged, you can continue
        with mcObj.runMCMC(nBurn=0), which discards the previous points and continues on.
            (Technically, it is burning all the previous samples, so it's a little misleading)
        
        Parameters
        ---------------------
        nBurn: int
            Number of samples to burn at the beginning. When nBurn is 0
            it will continue from the current state. When it is any other value
            it begins the mcmc sample from the initial guess point
        nRun: int
            The number of samples to run for
        """
        if nBurn != 0:
            self.pos, prob, state = self.sampler.run_mcmc(self.p0,nBurn)
            self.sampler.reset()
        else:
            self.sampler.reset()
        self.pos, prob, state = self.sampler.run_mcmc(self.pos,nRun)
        self.hasRun= True
        self.getResults()
    
    def getResults(self,lowPercent=15.9,highPercent=84.1):
        """ Get results from the chains in the sampler object """
        self.runCheck()
        lower,medianV,upper = [], [], []
        # As in CHIMERA, reject chains that are stuck in local min (at much lower likelihood)
        maxLogL = np.max(self.sampler.lnprobability)
        keepP = -self.sampler.lnprobability[:,-1] < -5. * maxLogL
        
        chainShape = self.sampler.chain.shape
        
        self.cleanFlatChain = np.reshape(self.sampler.chain[keepP,:,:],(np.sum(keepP)*chainShape[1],chainShape[2]))
        
        for i in range(self.ndim):
            flatChain = self.cleanFlatChain[:,i]
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
        self.getMaxL()
    
    def getMaxL(self):
        """ Get the maximum likelihood parameters """
        lnprob = self.sampler.lnprobability
        argMax = np.argmax(lnprob)
        tupArg = np.unravel_index(argMax,lnprob.shape)
        self.maxL = lnprob[tupArg]
        self.maxLparam = self.sampler.chain[tupArg]
    
    def showResult(self,showMedian=False,saveFile='plots/best_fit.pdf',
                   figsize=None,ax=None,fig=None):
        """ Shows the best-fit model from the median parameter values """
        self.runCheck()
        if showMedian == True:
            paramShow = self.results['Median']
        else:
            paramShow = self.maxLparam
        self.showGuess(showParamset=paramShow,saveFile=saveFile,figsize=figsize,ax=ax,fig=fig)
        print(self.chisquare(paramShow))
    
    def chisquare(self,param):
        """ Calculates the chi-squared"""
        chisquare = -2. * lnprob(param,self.x,self.y,self.yerr,self.model)
        dof = self.y.shape[0] - self.ndim
        return chisquare,chisquare/dof
    
    def runCheck(self):
        """A check that the MCMC has been run. If not, it runs"""
        if self.hasRun == False:
            print("Note: MCMC hasn't run yet and is now executing")
            self.runMCMC()
    
    def doCorner(self):
        """ Runs a corner plot """
        self.runCheck()
        
        fig = corner.corner(self.cleanFlatChain,labels=self.model.pnames,auto_bars=True,
                            quantiles=[0.159,0.841],show_titles=True)
        fig.savefig('plots/corner.pdf')
    
    def plotMCMC(self):
        """ Plots the MCMC histograms """
        self.runCheck()
        
        plt.close('all')
        fig, axArr = plt.subplots(1,self.ndim,figsize=(17,5))
        for i, ax in enumerate(axArr):
            postData = self.cleanFlatChain[:,i]
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
    
    def plotResids(self):
        """ Shows the residual time series """
        self.runCheck()
        self.showGuess(showParamset=self.maxLparam,residual=True,
                       saveFile='plots/residual_ML.pdf')

    def residHisto(self):
        """ Shows a histogram of the time series residuals """
        self.runCheck()
        yModel = self.model.evaluate(self.x,self.maxLparam)#self.guess)
        self.residy = self.y - yModel
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(self.residy, 30,normed=True,
                                   label='Residuals')
        
        ## Show a Gaussian dist
        sigGauss = np.nanstd(self.residy)
        yGauss = mlab.normpdf( bins, 0., sigGauss)
        l = plt.plot(bins, yGauss, 'r--', linewidth=2,label='Gaussian')
        ax.set_xlabel(self.yLabel)
        ax.set_ylabel('Probability')
        ax.legend(loc='best')
        fig.savefig('plots/histo_resids.pdf')

def show_k1255_results(fullRange=False):
    """ Show all KIC 1255 Results
    Parameters
    ------------
    fullRange: bool
        If true, show the full Y range. Otherwise it makes the range
        the same for all plots.
    """
    fileL = glob.glob('mcmcRuns/k1255_mod/kuiper/*.pic')
    totFigs = 6
    totFiles = len(fileL)
    #fig, axArr = plt.subplots(3,2,figsize=(12,8))
    
    if fullRange == True:
        wspace = 0.2
        fileDescrip = '_full_range'
    else:
        wspace = 0.0
        fileDescrip = ''
    
    fig = plt.figure(figsize=(12,8))
    nY, nX = (3,2)
    gs = gridspec.GridSpec(nY,nX,wspace=wspace,hspace=0.7)
    
    yArr, xArr = np.mgrid[0:nY,0:nX]
    yRavel = yArr.ravel()
    xRavel = xArr.ravel()
    
    for ind,oneFile in enumerate(fileL):
        mcObj = pickle.load(open(oneFile))
        #plotFile = 'plots/individual_fits/kic1255/best_fit{}.pdf'.format(mcObj.title)
        #ax = axArr.ravel()[ind]
        ax = fig.add_subplot(gs[ind])
        
        mcObj.showResult(fig=fig,ax=ax)
        ax.set_xlim(-0.28,0.28)
        
        if fullRange == True:
            if xRavel[ind] == 0:
                ax.set_ylabel('Norm Fl')
            else:
                ax.set_ylabel('')
        else:
            ax.set_ylim(0.99,1.01)
            if xRavel[ind] == 0:
                ax.set_ylabel('Norm Fl')
            else:
                ax.yaxis.set_visible(False)
                
        ## Save the best-fit model
        xModel = np.linspace(np.min(mcObj.xplot),np.max(mcObj.xplot),1024)
        yModel = mcObj.model.evaluate(xModel,mcObj.maxLparam)
        t = Table()
        t['Phase'] = xModel
        t['Best Fit Y'] = yModel
        
        t.write('tser_data/best_fit_models/kic1255/kic1255_best_fit_{}.csv'.format(mcObj.title),
                overwrite=True)
        
    
    outName = 'plots/individual_fits/kic1255/best_fit_all{}.pdf'.format(fileDescrip)
    fig.savefig(outName)

def run_k1255_nights():
    """ Runs the nights for KIC 1255 data and saves the results to Pickle files
    """
    
    photList = glob.glob('tser_data/refcor_phot/refcor_phot_kic1255_UT*.fits')
    
    mcmcDir = os.path.join('mcmcRuns','k1255_mod','kuiper')
    specDir = os.path.join('spectra','kic1255')
    for oneTSer in photList:
        mcObj = prepEmcee_k1255(photPath=oneTSer)
        
        print("Running MCMC for {}".format(mcObj.title))
        
        mcObj.runMCMC()
        
        ## Save the MCMC run
        emceePath = os.path.join(mcmcDir,'mcmc_'+mcObj.title+'.pic')
        
        pickle.dump(mcObj,open(emceePath,'w'))
        
        ## save the fit parameters and uncertainties
        resultPath = os.path.join(specDir,'fit_'+mcObj.title+'.csv')
        mcObj.results.write(resultPath,overwrite=True)
    

def prepEmcee_k1255(photPath='tser_data/refcor_phot/refcor_phot_kic1255_UT2016_06_12.fits'):
    """ Prepares Emcee for KIC 1255 data
    
    Parameters
    -----------
    """
    fullDat = Table.read(photPath)
    
    minValue, maxValue = 0.98, 1.02 ## clip for cosmic rays
    goodP = (fullDat['Y Corrected'] < maxValue) & (fullDat['Y Corrected'] > minValue)
    dat = fullDat[goodP]
    
    guess = [1.0, 0.0,0.0]
    spread = [0.5, 0.005,0.005]
    
    P = 0.6535538
    T0 = 2454833.039
    jd = dat['Time'] 
    phase = np.mod((jd - T0)/P, 1.0)
    highPt = phase > 0.5
    phase[highPt] = phase[highPt] - 1.0
    
    model = fit_models.kic1255Model()
    mcObj = oEmcee(model,phase,dat['Y Corrected'],dat['Y Corr Err'],
                   guess,spread,xLabel='Orbital Phase',yLabel='Normalized Flux',
                   title=dat.meta['NIGHT'])
                   
    return mcObj

def prepEmcee(nterms=1,moris=False,src='original1821',specWavel=1.08):
    """ Prepares Emcee for run 
    
    Example usage::
    mcObj = fit_tser_emcee.prepEmcee(nterms=1)
    mcObj.showResults()
    
    Parameters
    -------------
    nterms: int
        Passes to fSeries terms
    moris: bool
        Whether or not to look at MORIS photometry
    src: str
        Which source to look at, e.g. '0835' for 2MASS J08354256 0819237
    specWavel: float
        Which wavelength to look at for spectroscopy
    """
    if moris == True:
        dat = Table.read('tser_data/moris_1821_tser.fits')
        x = dat['hours']
        y = dat['AP00_RATIO_00_01']
        y1frac = dat['AP00_ERR_00']/dat['AP00_FLUX_00']
        y2frac = dat['AP00_ERR_01']/dat['AP00_FLUX_01']
        yerr = y * np.sqrt(y1frac**2 + y2frac**2)
    else:
        waveString = "{:.2f}".format(specWavel)
        tserFName = 'tser_data/'+src+'/timeser_'+waveString+'um_.txt'
        showTitle = 'Time Series at '+waveString+' $\mu$m'
        if os.path.exists(tserFName) == False:
            print("Unrecognized source")
            return 0
        
        if src == 'original1821':
            colNames = ['t','fl','flerr','model','resid']
        else:
            colNames = ['t','fl','flerr','model','resid','airmass']
        
        dat = ascii.read(tserFName,names=colNames)
        
        if 'airmass' in colNames:
            useAirmass = True
            x = np.zeros([2,len(dat)])
            x[0,:] = dat['t']
            x[1,:] = dat['airmass']
        else:
            x = np.array(dat['t'])
            useAirmass = False
        
        y = np.array(dat['fl'])
        ## For MORIS data, don't multiply
        if specWavel < 0.8:
            multFactor = 1.0
        else:
            multFactor = 4.0
            
        yerr = np.array(dat['flerr']) * multFactor
    
    
    model = fSeries(order=nterms,srcData=src,useAirmass=useAirmass)
    
    with open('parameters/fit_params.yaml') as paramFile:
        priorP = yaml.load(paramFile)
        periodGuess = priorP[src]['guess']['periodGuess']
    
    if useAirmass == True:
        if nterms == 1:
            guess = [periodGuess,0.,0.995,0.005,1.5,1.38]
            spread = [0.01,0.004,0.005,0.01,0.2,0.2]
        elif nterms == 2:
            guess = [periodGuess,0.,0.995,0.005,1.5,0.4,1.38,0.4]
            spread = [0.1,0.004,0.005,0.05,0.2,0.2,0.2,0.2]
        elif nterms == 3:
            guess = [periodGuess,0.,0.995,1.5,0.005,0.4,0.4,1.38,0.4,0.1]
            spread = [0.1,0.004,0.05,0.005,0.2,0.2,0.2,0.2,0.2,0.2]
        else:
            print("Doesn't accept nterms="+str(nterms))
    else:
        if nterms == 1:
            guess = [periodGuess,0.,0.995,1.5,1.38]
            spread = [0.01,0.004,0.01,0.2,0.2]
        elif nterms == 2:
            guess = [periodGuess,0.,0.995,1.5,0.4,1.38,0.4]
            spread = [0.1,0.004,0.05,0.2,0.2,0.2,0.2]
        elif nterms == 3:
            guess = [periodGuess,0.,0.995,1.5,0.4,0.4,1.38,0.4,0.1]
            spread = [0.1,0.004,0.05,0.2,0.2,0.2,0.2,0.2,0.2]
        else:
            print("Doesn't accept nterms="+str(nterms))
    
    mcObj = oEmcee(model,x,y,yerr,guess,spread,xLabel='Time (hr)',yLabel='Normalized Flux',
                   title=showTitle)
    
    return mcObj

def allBins(src='2mass_0835',wavelSearch=r"*"):
    """ Goes through each wavelength bin and does an MCMC fit, saving results
    """
    fileList = glob.glob('tser_data/'+src+'/'+wavelSearch+'.txt')
    
    specDir = os.path.join('spectra',src)
    ## Make a directory for spectra if there isn't one yet
    chisqDir = os.path.join('chisquare',src)
    plotDir = os.path.join('plots','individual_fits',src)
    mcmcDir = os.path.join('mcmcRuns','fSeries',src)
    for onePath in [specDir,chisqDir,plotDir,mcmcDir]:
        if os.path.exists(onePath) == False:
            os.mkdir(onePath)
    
    
    for oneFile in fileList:
        baseName = os.path.basename(oneFile)
        thisWave = float(baseName.split("_")[1].split("um")[0])
        
        waveString = "{:.2f}".format(thisWave)
        print('Working on '+waveString)
        ## Run the MCMC sampler
        mcObj = prepEmcee(src=src,specWavel=thisWave,nterms=2)
        mcObj.runMCMC()
        
        emceePath = os.path.join(mcmcDir,'mcmc_'+waveString+'.pic')
        
        ## Save the results
        mcObj.results.write(specDir+'/fit_'+waveString+'.csv')
        pickle.dump(mcObj,open(emceePath,'w'))
        
        chisq, chisqpDOF = mcObj.chisquare(mcObj.maxLparam)
        t2 = Table()
        t2['Chisquare Best'] = [chisq]
        t2['Chisquare/DOF Best'] = [chisqpDOF]
        t2.write(chisqDir+'/chisq_'+waveString+'.csv')
        mcObj.showResult(saveFile=os.path.join(plotDir,'tser_'+waveString+'.pdf'))
        
        
        

class getSpectrum():
    """ A spectrum object that gathers spectra from individual wavelength fits"""
    def __init__(self,src):
        self.src = src
        self.fileList = glob.glob('spectra/'+src+'/fit*.csv')
        self.plotPath = os.path.join('plots','spectra',src)
        if os.path.exists(self.plotPath) == False:
            os.mkdir(self.plotPath)
        exampleFile = ascii.read(self.fileList[0])
        self.paramList = exampleFile['Parameter']
    
    
    def getSpectrum(self,oneParam,timeVersion=False):
        """ Gathers a spectrum for a given file List of wavelength bin CSV files
        Parameters
        ----------
        oneParam: string
            Parameter name in model
        timeVersion: bool
            If true, it will look at time variations
            Otherwise, it's organized by wavelength
        """
        lowLim, medLim, hiLim = [], [], []
        wavel = []
    
        for oneFile in self.fileList:
            t = ascii.read(oneFile)
            AmpRow = t['Parameter'] == oneParam
            lowLim.append(float(t['Lower'][AmpRow]))
            medLim.append(float(t['Median'][AmpRow]))
            hiLim.append(float(t['Upper'][AmpRow]))
            baseName = os.path.basename(oneFile)
            if timeVersion == True:
                thisWavel = os.path.splitext(baseName)[0]
            else:
                thisWavel = float(os.path.splitext(baseName.split("_")[1])[0])
            wavel.append(thisWavel)
    
        
        yerrLow = np.array(medLim) - np.array(lowLim)
        yerrHigh = np.array(hiLim) - np.array(medLim)
    
        specTable = Table()
        if timeVersion == True:
            specTable['Time'] = wavel
        else: 
            specTable['Wavel'] = wavel
        specTable['Median'] = medLim
        specTable['yerrLow'] = yerrLow
        specTable['yerrHigh'] = yerrHigh
        specTable['yerrAverage'] = (yerrLow + yerrHigh)/2.
        return specTable
    
    def plotSpectrum(self,oneParam,showComparison=True,ax=None,fig=None,
                     legLabel=None):
        """ Plots the spectrum for a given parameter
        
        Parameters
        --------------
        showComparison: bool
             show comparison to previous spectra
        """
        
        t = self.getSpectrum(oneParam)
        
        plt.close('all')
        if ax == None:
            self.fig, self.ax = plt.subplots(figsize=(6,4))
        else:
            self.fig, self.ax = fig, ax
        self.ax.errorbar(t['Wavel'],t['Median'],fmt="o",yerr=[t['yerrLow'],t['yerrHigh']],
                         label=legLabel)
        with open('parameters/fit_params.yaml') as paramFile:
            priorP = yaml.load(paramFile)
            if 'spectraYrange' in priorP[self.src] and oneParam == r'A$_1$':
                self.ax.set_ylim(priorP[self.src]['spectraYrange'])
                customYLabel = r'Amplitude (%)'
                
            else:
                customYLabel = oneParam
        
        
        self.ax.set_xlabel('Wavelength ($\mu$m)')
        self.ax.set_ylabel(customYLabel)
        self.fig.savefig(os.path.join(self.plotPath,oneParam+'_spectrum.pdf'),bbox_inches='tight')
    

def plotSpectra(src='2mass_0835'):
    """ Goes through each wavelength bin and plots the spectra
    """
    
    specObj = getSpectrum(src)
    
    for oneParam in specObj.paramList:
        specObj.plotSpectrum(oneParam)


def flatLineTest(src='2mass_1821',param='t$_1$',comparePhase=False,waverage=False,
                waveRange=None):
    """ Tests if the phase offset is consistent with a flat line
    Parameters
    ---------------
    src: str
        Source target
    param: str
        Parameter (for example r"t$_1"$)
    comparePhase: bool
        Plots offset versus amplitude
    waverage: bool
        Instead of doing a flat line test, find the weighted average of the 
        given parameter
    """
    specObj = getSpectrum(src)
    t = specObj.getSpectrum(param)
    if waveRange is not None:
        ind = (t['Wavel'] > waveRange[0]) & (t['Wavel'] < waveRange[1])
        t = t[ind]
    
    weights = 1./t['yerrAverage']**2
    avg = np.sum(t['Median'] * weights)/np.sum(weights)
    
    chisQ = np.sum((t['Median'] - avg)**2 / t['yerrAverage']**2)
    dof = len(t) - 1
    chisQperDOF = chisQ / dof
    print('Chi-squared/dof = ',chisQperDOF)
    
    if comparePhase == True:
        t2 = specObj.getSpectrum('A$_1$')
        fig, ax = plt.subplots()
        ax.errorbar(t['Median'],t2['Median'],yerr=[t2['yerrLow'],t2['yerrHigh']],
                    xerr=[t['yerrLow'],t['yerrHigh']],fmt='o')
        ax.set_xlabel('Phase Offset (hr)')
        ax.set_ylabel('Amplitude (%)')
        fig.savefig(os.path.join(specObj.plotPath,'offset_vs_amplitude.pdf'))
    elif waverage == True:
        sigmaAvg = 1./np.sqrt(np.sum(weights))
        print('Average '+param+':')
        print(avg,' +/- ',sigmaAvg)
    else:
        specObj.plotSpectrum(param)
        specObj.ax.plot(t['Wavel'],avg * np.ones(len(t)),label='Flat')
        specObj.fig.savefig(os.path.join(specObj.plotPath,'offset_spectrum_w_avg.pdf'))

def compareMultiTerms(maxTerms = 3):
    """ Compares a single versus multiple cosine terms fit
    
    Parameters
    -------------
    maxterms: int
        The maximum terms to show of the cosine fits to the time series
    """
    mcArray = []
    for i in range(maxTerms):
        thisMC = prepEmcee(nterms=i+1)
        thisMC.runMCMC()
        mcArray.append(thisMC)
    
    fig, ax = plt.subplots()
    ax.errorbar(thisMC.x,thisMC.y,yerr=thisMC.yerr,fmt='o')
    xmodel = np.linspace(np.min(thisMC.x),np.max(thisMC.x),1024)
    for i in range(maxTerms):
        thisMC = mcArray[i]
        chisq, chisPDof = thisMC.chisquare(thisMC.maxLparam)
        thisLabel = str(i+1)+' Term Fit, $\chi^2$/dof={:.1f}'.format(chisPDof)
        yShow = thisMC.model.evaluate(xmodel,thisMC.maxLparam)
        ax.plot(xmodel,yShow,linewidth=3.,label=thisLabel)
    
    ax.set_xlabel(thisMC.xLabel)
    ax.set_ylabel(thisMC.yLabel)
    ax.set_title(thisMC.title)
    
    ax.legend(loc='best')
    fig.savefig('plots/best_fit_comparison.pdf')
    return mcArray

def showTDiff(paramVary='temp',pset='normal'):
    """ Shows some example TDiff models """
    mc = prepEmceeSpec()
    if pset == 'normal':
        paramset = [0,0.05,1450,1400]
    else:
        paramset = [0,1.5e-3,1800,1400]
    
    nparam = len(paramset)
    fig, ax = plt.subplots()
    ax.errorbar(mc.xplot,mc.y,yerr=mc.yerr,fmt='o')
    ax.set_xlabel(mc.xLabel)
    ax.set_ylabel(mc.yLabel)
    
    #mc.showGuess(showParamset=paramset,fig=fig,ax=ax)
    xtemp = np.linspace(0.8,2.4,1024)
    
    if paramVary == 'temp':
        paramInd = 2
        if pset == 'normal':
            paramTry = [1425,1450,1475,1500]
        else:
            paramTry = np.linspace(1500,2000,5)
    elif paramVary == 'alpha_1':
        paramInd = 0
        if pset == 'normal':
            paramTry = np.linspace(0,0.9,5)
        else:
            paramTry = np.linspace(0,0.4,5)
    else:
        paramVary = 'beta'
        paramInd = 1
        if pset == 'normal':
            paramTry = np.linspace(0,0.1,5)
        else:
            paramTry = np.linspace(0,4e-3,5)
    
    for oneValue in paramTry:
        paramset[paramInd] = oneValue
        
        y = mc.model.evaluate(xtemp,paramset)
        ax.plot(xtemp,y,label=mc.model.pnames[paramInd]+'='+str(oneValue))
    ax.legend()
    titleStr = ''
    for oneParam in np.arange(nparam):
        if oneParam != paramInd:
            if titleStr != '':
                titleStr = titleStr + ','
            titleStr = titleStr + mc.model.pnames[oneParam]+'='
            titleStr = titleStr + str(paramset[oneParam])
            
    ax.set_title(titleStr)
    if pset == 'normal':
        outName = 'normal_'+paramVary
    else:
        outName = 'extreme_'+paramVary
        
    fig.savefig('tser_data/model_parameter_explorations/'+outName+'_varied.pdf')
    fig.show()

def prepEmceeSpec(method='tdiff',logNorm=True,useIDLspec=False,src='2mass_1821',
                  variableSigma=False,ampParameter=r"A$_1$"):
    """ Prepares Emcee run for fitting spectra
    
    Parameters
    ------------
    method: str
        Method of fitting - Temp Diff vs Mie Scattering
    """
    
    if useIDLspec == True:
        dat = ascii.read('tser_data/amp_vs_wavl.txt')
        x = np.array(dat['Wavelength(um)'])
        y = np.array(dat['Amp']) * 100.
        yerr = np.array(dat['Amp_Err']) * 100.        
    else:
        specObj = getSpectrum(src)
        t = specObj.getSpectrum(ampParameter)
        x = np.array(t['Wavel'])
        y = np.array(t['Median'])
        yerr = np.array(t['yerrAverage'])
    
    if method == 'tdiff':
        # Temperature difference model
        model = tdiffModel()
        guess = [0.0013,1e-4,2300,1500]
        spread = [0.001,5e-5,200,200]
    elif method == 'mie':
        model = mieModel(logNorm=logNorm,variableSigma=variableSigma)
        if variableSigma == True:
            guess = [0.5, 0.2, 0.5]
            spread = [0.1, 0.1, 0.1]
        else:
            guess = [0.5,0.2]
            spread = [0.1,0.1]
    else:
        print('Unrecognized model')
        return 0

    
    mcObj = oEmcee(model,x,y,yerr,guess,spread,xLabel='Wavelength ($\mu$m)',
                   yLabel='Amplitude (%)')
    return mcObj

def tser_plots(src='2mass_1821',removeBaselines=True,abbreviated=False,
               showTelluric=False):
    """ Shows the time series for each wavelength
    Parameters
    ---------------------
    src: str
        Source name
    removeBaselines: bool
        Take out the baselines fit to the light curves
    abbreviated: bool
        Show a smaller y range
    showTelluric: bool
        Show the regions where telluric absorption is strong
    """
    
    
    fileList = glob.glob('mcmcRuns/fSeries/'+src+'/*.pic')
    if abbreviated == True:
        newList = np.array(fileList)
        fileList = newList[np.mod(np.arange(len(fileList)),3) == 2]
    
    plt.close('all')
    if abbreviated == True:
        fig, ax = plt.subplots(figsize=(7,5))
    else:
        fig, ax = plt.subplots(figsize=(7,10))
    
    ## Create a directory for baseline-removed time series
    ## These are useful for Theodora's spot models
    
    cleanDir = os.path.join('cleaned_tser',src)
    if os.path.exists(cleanDir) == False:
        os.mkdir(cleanDir)
    
    if src == '2mass_1821':
        xLabel='JD - 2457565 (hr)'
    elif src == '2mass_0835':
        xLabel='JD - 2457388 (hr)'
    else:
        xLabel = 'Time'
    
    offset = 0.035
    waveList = []
    
#    for waveInd,oneFile in enumerate(fileList):
    for waveInd,oneFile in enumerate(fileList):
        baseName = os.path.basename(oneFile)
        basePrefix = os.path.splitext(baseName)
        thisWave = float(basePrefix[0].split("_")[1].split("um")[0])
        waveString = "{:.2f}".format(thisWave)
        waveList.append(thisWave)
        
        mcObj = pickle.load(open(oneFile))
        yModel = mcObj.model.evaluate(mcObj.x,mcObj.maxLparam)
        #mcObj = prepEmcee(src=src,specWavel=thisWave,nterms=2)
        if removeBaselines == True:
            ## Findi the model with baselines
            paramBaseline = deepcopy(mcObj.maxLparam)
            paramBaseline[mcObj.model.Aind] = 0.
            yBaseline = mcObj.model.evaluate(mcObj.x,paramBaseline)
            
            yShow = mcObj.y - yBaseline + 1.
            
            ## Save the time series
            t = Table()
            t[xLabel] = mcObj.xplot
            t['Norm Flux'] = np.array(yShow)
            t['Flux Err'] = np.array(mcObj.yerr)
            cleanFile = os.path.join(cleanDir,'tser_'+waveString+'.txt')
            ascii.write(t,cleanFile,delimiter=' ',overwrite=True)
            
            yModel = yModel - yBaseline + 1.
        else:
            yShow = mcObj.y
        
        plotPoints = ax.plot(mcObj.xplot,yShow - offset * waveInd,'o',rasterized=True,
                             markersize=4.)
        
        ax.plot(mcObj.xplot,yModel- offset * waveInd,linewidth=2.,rasterized=True)
        
        ax.text(np.max(mcObj.xplot)+0.3,
                np.median(yShow - offset * waveInd),waveString+'$\mu$m',
                color=plotPoints[0].get_color(),rasterized=True)
    
    if abbreviated == True:
        ax.set_ylim(0.7,1.07)
    else:
        ax.set_ylim(0.14,1.1)
    ax.set_xlim(np.min(mcObj.x) - 0.5,np.max(mcObj.x)+1.8)
    
    if showTelluric == True:
        with open('parameters/telluric_bands.yaml') as tellFile:
            telluricData = yaml.load(tellFile)
        waveList = np.array(waveList)
        diff = np.diff(waveList)
        waveWidths = np.insert(diff,0,diff[0])
        waveStarts = waveList - waveWidths
        waveEnds = waveList + waveWidths
        
        nWaves = len(waveList)
        waveIndices = np.arange(nWaves)
        
        for oneLine in telluricData['SpeX IRTF']:
            minWaveInd = np.min(waveIndices[waveEnds >= oneLine[0]])
            maxWaveInd = np.max(waveIndices[waveStarts <= oneLine[1]])
            minWavePos = (minWaveInd - 0.4) * offset
            maxWavePos = (maxWaveInd + 0.4) * offset
            xLimits = ax.get_xlim()
            xGap = np.mean([xLimits[0],np.min(mcObj.xplot)])
            quartGap = (xGap - xLimits[0])* 0.25 ## size of bracket tick
            bracketX = [xGap + quartGap,xGap,xGap, xGap + quartGap]
            bracketY = 1. - np.array([minWavePos,minWavePos,maxWavePos,maxWavePos])
            plt.plot(bracketX,bracketY,linewidth=3,
                     color='green')
    
    ax.set_xlabel(xLabel)
    ax.set_ylabel('Flux Ratio + Offset')
    fig.savefig('plots/'+src+'_tser.pdf',bbox_inches='tight')
    
def checkCleaned(src='2mass_1821'):
    """ Check the cleaned time series """
    fileL = glob.glob(os.path.join('cleaned_tser',src,'tser*.txt'))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7,10))
    offset = 0.03
    for ind, oneFile in enumerate(fileL):
        t = ascii.read(oneFile)
        tKey = 'JD - 2457565 (hr)'
        baseName = os.path.basename(oneFile)
        basePrefix = os.path.splitext(baseName)
        waveString = basePrefix[0].split("_")[1]
        
        bars = ax.errorbar(t[tKey],t['Norm Flux'] - offset * ind,yerr=t['Flux Err'])
        
        ax.text(np.max(t[tKey]) + 0.2,np.median(t['Norm Flux']) - offset * ind,
                       waveString,color=bars[0].get_color())
    
    ax.set_xlabel(tKey)
    ax.set_ylabel('Normalized Flux')
    ax.set_xlim(np.min(t[tKey]) - 0.2,np.max(t[tKey]) + 1.)
    fig.savefig('plots/tser_w_errors.pdf')
    

def showDistributions(src='2mass_1821',fig=None,ax=None):
    """ Shows the particle size distributions from the fit """
    mcObj = pickle.load(open('mcmcRuns/mie_model/2mass_1821_mcmc_free_sigma.pic'))
    totChains = mcObj.cleanFlatChain.shape[0]
    randSet = np.random.randint(totChains,size=150)
    #plt.close('all')
    if ax == None:
        fig, ax = plt.subplots()
    lineData = []
    for oneSet in randSet:
        b, rad, sig = mcObj.cleanFlatChain[oneSet,:]
        x, y, dx = mie_model.xyLogNorm(sig,rad)
        thisLine = []
        for ind in np.arange(x.size):
            thisLine.append((x[ind],y[ind]))
        lineData.append(thisLine)
        #ax.plot(x,y,rasterized=True,alpha=1.0,color='blue',linewidth=1.)
    lines = LineCollection(lineData,linewidths=1,alpha=0.2)
    ## Show the maximum Likelihood line
    x, y, dx = mie_model.xyLogNorm(mcObj.maxLparam[2],mcObj.maxLparam[1])
    ax.plot(x,y,color='red',linewidth=3)
    ax.add_collection(lines)
    ax.set_xlim(0,0.8)
    ax.set_ylim(0,15)
    ax.set_xlabel('Radius ($\mu$m)')
    ax.set_ylabel('Relative Number')
    fig.canvas.draw()
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
    fig.show()
    fig.savefig('plots/rad_distribution.pdf',bbox_inches='tight')

def distributionCorner(src='2mass_1821'):
    """ 
    Plots the MCMC corner plot and as an inset shows the particle size distribution
    """
    mcObj = pickle.load(open('mcmcRuns/mie_model/2mass_1821_mcmc_free_sigma.pic'))
    
    labels = deepcopy(mcObj.model.pnames)
    labels[0] = 'K'
    labels[2] = '$\sigma_s$'
    
    fig = corner.corner(mcObj.cleanFlatChain,labels=labels,auto_bars=True,
                        quantiles=[0.159,0.841],show_titles=True)
    ax2 = fig.add_axes([0.7,0.7,0.3,0.3])
    showDistributions(fig=fig,ax=ax2)
    fig.show()

def bdPaperSpecFits(src='2mass_1821',abbreviated=False,
                    ampParameter=r"A$_1$",phaseParameter=r"t$_1$"):
    """ Makes the spectral fits for the Brown Dwarf paper 
    Parameters
    -------------
    src: str
        Source. For initial paper, either '2mass_1821' or '2mass_0835'
    abbreviated: bool
        If true, only show the top panel. Good for presentations/proposals
    ampParameter: str
        Amplitude parameter. Options include r"A$_1$" and r"A$_2$"
    phaseParameter: str
        Phase offset parameter. Options include 
    """
    emceeDir = 'mcmcRuns/mie_model/'+src+'_mcmc_free_sigma.pic'
    if os.path.exists(emceeDir) == False:
        mcObj = prepEmceeSpec(method='mie',src=src)
        mcObj.runMCMC()
        mcObj.runMCMC(nBurn=0)
        pickle.dump(mcObj,open(emceeDir,'w'))
    else:
        mcObj = pickle.load(open(emceeDir))
    
    if abbreviated == True:
        fig, ax1 = plt.subplots(figsize=(7,5))
    else:
        fig, (ax1,ax2) = plt.subplots(2,figsize=(6,7),sharex=True)
    
    
    ## Make a spectrum object
    specObj = getSpectrum(src)
    specObj.plotSpectrum(ampParameter,ax=ax1,fig=fig,legLabel='IRTF Amp')
    
    ## Show the model
    specTable = specObj.getSpectrum(ampParameter)
    xmodel = np.linspace(np.min(specTable['Wavel']),
                         np.max(specTable['Wavel']),512.)
    if src == '2mass_0835':
        ## Show a Zero amplitude line for 2MASS 0835 since we don't find any significant variability
        ymodel = np.zeros_like(xmodel)
        modelLabel = ampParameter+'=0'
    else:
        ymodel = mcObj.model.evaluate(xmodel,mcObj.maxLparam)
        #label
        radString = str(np.round(mcObj.maxLparam[1],2))
        modelLabel = 'Mie r='+radString+' $\mu$m'
    
    ## Only show the model for A1
    if ampParameter == r"A$_1$":
        ax1.plot(xmodel,ymodel,color='green',linewidth=3.,label=modelLabel)
    ax1.set_ylabel(ampParameter+' (%)')
    
    ax1.set_ylim(-0.9,2.)
    if (src == '2mass_1821') & (ampParameter == r"A$_1$"):
        ## Show the HST amplitude
        otherDat = ascii.read('existing_dat/spectra/fratio_yang2016.csv')
        YangAmp = (otherDat['fratio'] - 1.)/(otherDat['fratio'] + 1.)
        ax1.plot(otherDat['wavelength'],YangAmp * 100.,color='black',label='WFC3 Amp')
        #ax1.text(np.mean(otherDat['wavelength']),np.mean(YangAmp * 100.)*1.05,
        #             'WFC3 amp')
    ax1.legend(loc='best',frameon=False)
    
    ## Show the phase spectrum
    if abbreviated == False:
        specObj.plotSpectrum(phaseParameter,ax=ax2,fig=fig,legLabel='')
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        ax2.set_ylabel(phaseParameter+' (hr)')
        axList = [ax1,ax2]
    else:
        axList = [ax1]
    
    ## Show the telluric-affected regions
    with open('parameters/telluric_bands.yaml') as tellFile:
        telluricData = yaml.load(tellFile)
    
    for oneAx in axList:
        for oneLine in telluricData['SpeX IRTF']:
            oneAx.axvspan(oneLine[0],oneLine[1],alpha=0.2,color='green')
    
    fig.savefig('plots/best_fit_'+src+'.pdf',bbox_inches="tight")
    
    mcObj.doCorner()
    
    
    return mcObj

def roughCovar(src='2mass_1821',fIndex=0):
    """ Calculates an approximate covariance matrix.
    Uses medians instead of mean so that it's more robust 
    """
    mcmcDir = os.path.join('mcmcRuns','fSeries',src)
    fileL = glob.glob(os.path.join(mcmcDir,'*.pic'))
    oneFile = fileL[fIndex]
    mcObj = pickle.load(open(oneFile))
    covR = np.zeros([mcObj.ndim,mcObj.ndim])
    covTable = Table()
    covTable['Parameter'] = mcObj.model.pnames
    for i in range(mcObj.ndim):
        for j in range(mcObj.ndim):
            iData = mcObj.cleanFlatChain[:,i]
            jData = mcObj.cleanFlatChain[:,j]
            covR[j,i] = np.median((iData - np.median(iData)) * (jData - np.median(jData)))
        covTable[mcObj.model.pnames[i]] = covR[:,i]
    return covTable, covR
    
def checkAllCov(src='2mass_1821',paramNums=[4,5]):
    mcmcDir = os.path.join('mcmcRuns','fSeries',src)
    fileL = glob.glob(os.path.join(mcmcDir,'*.pic'))
    for oneIndex in range(len(fileL)):
        covTable, covR = roughCovar(src=src,fIndex=oneIndex)
        mcObj = pickle.load(open(fileL[oneIndex]))
        name1 = mcObj.model.pnames[paramNums[0]]
        name2 = mcObj.model.pnames[paramNums[1]]
        print(oneIndex, fileL[oneIndex])
        print('Var ',name1,covR[paramNums[0],paramNums[0]])
        print('Var ',name2,covR[paramNums[1],paramNums[1]])
        print('Co-Var ',name1,' ',name2,covR[paramNums[0],paramNums[1]])
        print(' ')
        
