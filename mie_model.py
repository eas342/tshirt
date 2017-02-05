import numpy as np
import miescatter
import pdb
import matplotlib.pyplot as plt

def extinct(wavel,rad=1.0,n=complex(1.825,-1e-4),logNorm=False,
            npoint=100):
    """
    Calculates the Mie extinction cross section Q_ext as a function of wavelength
    
    Arguments
    ------------------
    rad: float
        The radius of the particle
    wavel: float,arr
        The array of wavelengths to evaluate
    """
    sz = 2. * np.pi * rad/np.array(wavel)
    
    if logNorm == True:
        ## Generate an array of points along the distribution
        ## Size multiplier
        nwav = sz.size
        ## Size to evaluate lognormal weights
        sizeEval = np.linspace(0.2,5.,npoint)
        weights = lognorm(sizeEval,0.5,1.)
        ## Arrange the array into 2D for multiplication of the grids
        sizeMult = (np.tile(sizeEval,(nwav,1))).transpose()
        sizeArr = np.tile(sz,(npoint,1))
        
        eval2D = sizeMult * sizeArr
        finalEval = eval2D.ravel() ## Evaluate a 1D array
        
    else:
        finalEval = np.array(sz)
        
    ## Make all points with sz > 100. the same as 100
    ## Otherwise the miescatter code quits without any warning or diagnostics
    highp = finalEval > 100.
    finalEval[highp] = 100.
    
    qext,qsca,qabs,qbk,qpr,alb,g = miescatter.calcscatt(n,finalEval,n=finalEval.size)
    
    if logNorm == True:
        ## Now sum by the weights
        qext2D = np.reshape(qext,(npoint,nwav))
        finalQext = np.dot(weights,qext2D) * (sizeEval[1] - sizeEval[0])
    else:
        finalQext = qext
        
    return finalQext

def lognorm(x,s,med):
    """
    Calculates a log-normal size distribution
    
    Arguments
    ------------------
    x: arr
        The input particle size
    s: float
        The sigma value
    med: float
        The median particle size
    """
    mu = np.log(med)
    y = 1. / (s*x*np.sqrt(2.*np.pi)) * np.exp(-0.5*(np.log(x-mu)/s)**2)
    return y
    
def compareTest():
    """
    Compares the Single Particle and Log-normal distributions
    """
    x = np.linspace(0.9,10,1024)
    rad = 1.0
    ysingle = extinct(x,rad=rad,logNorm=False)
    plt.plot(x,ysingle,label='Single Particle')
    npointA = [10,32,64,1024]
    for npoint in npointA:
        ymulti = extinct(x,rad=rad,logNorm=True,npoint=npoint)
        plt.plot(x,ymulti,label='Log Normal N='+str(npoint))
    plt.legend()
    plt.show()
    