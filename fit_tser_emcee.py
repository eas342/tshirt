import numpy as np
import emcee
import pdb
import os
import glob
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import corner
import yaml

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
        The indices for the 
    """
    def __init__(self,srcData=None,order=2):
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
        for Amp,freq,t in zip(p[self.Aind],freqArr,p[self.tind]):
            cosTerms += Amp/100. * np.cos(freq * 2. * np.pi * (x - t)/p[0])
        baseLine = p[1] * x + p[2]
        
        return cosTerms + baseLine
    
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
    
    def showGuess(self,showParamset=None,saveFile=None,ax=None,fig=None,
                  residual=False):
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
            fig, ax = plt.subplots()
        
        if showParamset is None:
            modelParam = self.guess
        else:
            modelParam = showParamset
        xmodel = np.linspace(np.min(self.x),np.max(self.x),1024)
        
        if residual == True:
            yModel = self.model.evaluate(self.x,modelParam)
            yShow = self.y - yModel
            print('Median Error = '+str(np.median(self.yerr)))
            print('Standard Dev Residuals = '+str(np.nanstd(yShow)))
            ax.errorbar(self.x,yShow,yerr=self.yerr,fmt='o')
        else:
            ax.errorbar(self.x,self.y,yerr=self.yerr,fmt='o')
            yModel = self.model.evaluate(xmodel,modelParam)
            yShow = yModel
            ax.plot(xmodel,yShow,linewidth=3.)
        ax.set_xlabel(self.xLabel)
        ax.set_ylabel(self.yLabel)
        ax.set_title(self.title)
        if saveFile is None:
            fig.show()
        else:
            fig.savefig(saveFile)
    
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
        self.getMaxL()
    
    def getMaxL(self):
        """ Get the maximum likelihood parameters """
        lnprob = self.sampler.lnprobability
        argMax = np.argmax(lnprob)
        tupArg = np.unravel_index(argMax,lnprob.shape)
        self.maxL = lnprob[tupArg]
        self.maxLparam = self.sampler.chain[tupArg]
    
    def showResult(self,showMedian=False,saveFile='plots/best_fit.pdf'):
        """ Shows the best-fit model from the median parameter values """
        self.runCheck()
        if showMedian == True:
            paramShow = self.results['Median']
        else:
            paramShow = self.maxLparam
        self.showGuess(showParamset=paramShow,saveFile=saveFile)
        print(self.chisquare(paramShow))
    
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
        dat = ascii.read(tserFName,
                         names=['t','fl','flerr','model','resid'])    
        x = np.array(dat['t'])
        y = np.array(dat['fl'])
        yerr = np.array(dat['flerr']) * 4.0
    
    
    model = fSeries(order=nterms,srcData=src)
    
    with open('parameters/fit_params.yaml') as paramFile:
        priorP = yaml.load(paramFile)
        periodGuess = priorP[src]['guess']['periodGuess']
    
    if nterms == 1:
        guess = [periodGuess,0.,0.995,1.5,1.38]
        spread = [0.01,0.004,0.05,0.2,0.2]
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

def allBins(src='2mass_0835'):
    """ Goes through each wavelength bin and does an MCMC fit, saving results
    """
    fileList = glob.glob('tser_data/'+src+'/*.txt')
    ## Make a directory for spectra if there isn't one yet
    specDir = os.path.join('spectra/',src)
    chisqDir = os.path.join('chisquare',src)
    plotDir = os.path.join('plots','individual_fits',src)
    for onePath in [specDir,chisqDir,plotDir]:
        if os.path.exists(onePath) == False:
            os.mkdir(onePath)
    
    for oneFile in fileList:
        baseName = os.path.basename(oneFile)
        thisWave = float(baseName.split("_")[1].split("um")[0])
        mcObj = prepEmcee(src=src,specWavel=thisWave)
        mcObj.runMCMC()
        waveString = "{:.2f}".format(thisWave)
        mcObj.results.write(specDir+'/fit_'+waveString+'.csv')
        
        chisq, chisqpDOF = mcObj.chisquare(mcObj.maxLparam)
        t2 = Table()
        t2['Chisquare Best'] = [chisq]
        t2['Chisquare/DOF Best'] = [chisqpDOF]
        t2.write(chisqDir+'/chisq_'+waveString+'.csv')
        mcObj.showResult(saveFile=os.path.join(plotDir,'tser_'+waveString+'.pdf'))

def plotSpectra(src='2mass_0835',param="A$_1$"):
    """ Goes through each wavelength bin and plots the spectra
    """
    fileList = glob.glob('spectra/'+src+'/fit*.csv')
    lowLim, medLim, hiLim = [], [], []
    wavel = []
    for oneFile in fileList:
        t = ascii.read(oneFile)
        AmpRow = t['Parameter'] == param
        lowLim.append(t['Lower'][AmpRow])
        medLim.append(t['Median'][AmpRow])
        hiLim.append(t['Upper'][AmpRow])
        baseName = os.path.basename(oneFile)
        thisWavel = float(os.path.splitext(baseName.split("_")[1])[0])
        wavel.append(thisWavel)
    yerrLow = np.array(medLim) - np.array(lowLim)
    yerrHigh = np.array(hiLim) - np.array(medLim)
    plt.close('all')
    fig, ax = plt.subplots()
    ax.errorbar(wavel,medLim,fmt="o",yerr=[yerrLow,yerrHigh])
    with open('parameters/fit_params.yaml') as paramFile:
        priorP = yaml.load(paramFile)
        if 'spectraYrange' in priorP[src]:
            ax.set_ylim(priorP[src]['spectraYrange'])
    
    ax.set_xlabel('Wavelength ($\mu$m)')
    ax.set_ylabel(param)
    fig.savefig('plots/'+src+"_spectrum.pdf")

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

def prepEmceeSpec():
    """ Prepares Emcee run for """
    model = tdiffModel()
    
    dat = ascii.read('tser_data/amp_vs_wavl.txt')
    x = np.array(dat['Wavelength(um)'])
    y = np.array(dat['Amp']) * 100.
    yerr = np.array(dat['Amp_Err']) * 100.
    guess = [0.0013,1e-4,2300,1500]
    spread = [0.001,5e-5,200,200]
    
    mcObj = oEmcee(model,x,y,yerr,guess,spread,xLabel='Wavelength ($\mu$m)',
                   yLabel='Amplitude (%)')
    return mcObj



