import time
import fit_tser_emcee
import numpy as np

def calc_time(testObject,nRepeats=1,units='ms'):
    """
    Calculates how long it takes to evaluate a given method
    
    Expects an object with a method called evaluate
    """
    tArr = []
    for i in range(nRepeats):
        t1 = time.time()
        outDump = testObject.evaluate()
        t2 = time.time()
        tArr.append(t2-t1)
    avg = np.mean(tArr)
    if units == 'ms':
        showVal = avg * 1000.
    elif units == 's':
        showVal = avg
    else:
        print('Unrecognized units')
        showVal = 0
    print('Average time= '+str(showVal)+' '+units)

class fSeriesEval():
    """
    Evaluation of a Fourier series model
    """
    def __init__(self):
        self.fSmodel = fit_tser_emcee.fSeries()
        self.x = np.linspace(0,5,1024)
        self.p = np.array([4.1,0.,0.995,1.5,0.4,0.4,1.38,0.4,0.1])
        
    def evaluate(self):
        return self.fSmodel.evaluate(self.x,self.p)

def time_fSeriesEval():
    f = fSeriesEval()
    calc_time(f,nRepeats=30)
    
class run_mcmc_3terCos():
    """ 
    Runs a 3 term MCMC fit
    """
    def __init__(self):
        self.mcObj = fit_tser_emcee.prepEmcee(nterms=3)
    
    def evaluate(self):
        return self.mcObj.runMCMC()

def time_mcmc3term():
    """ Times the 3 term cosine model MCMC """
    ## Took 0.09 ms for summation
    f = run_mcmc_3terCos()
    calc_time(f,nRepeats=1,units='s')
    