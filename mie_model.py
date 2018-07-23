import numpy as np
import warnings
try:
    import miescatter
except ImportError:
    warnings.warn("Unable to import miescatter, so scattering models won't work")
import pdb
import matplotlib.pyplot as plt
try:
    import es_gen
except ImportError:
    warnings.warn("Unable to import es_gen. Some of the Mie scattering models may not work")
import yaml

coeff = yaml.load(open('parameters/mie_params/simplePoly.yaml'))


def polyExtinct(wavel,rad=1.0,type=r'Simple n=(1.67-0.006j)'):
    """
    Extinction function using a high order polynomial fit
    """
    normWave = wavel/rad
    Carr = coeff[type]['coefficients']
    return np.polyval(Carr,normWave)

def polyExtinctMatrix(wavel,rad=1.0,type=r'Simple n=(1.67-0.006j)'):
    """
    Extinction function using a high order polynomial fit
    """
    normWave = wavel/rad
    Carr = np.array(coeff[type]['coefficients'])
    nOrder = len(Carr)
    #nX = normWave.size[0]
    
    x2D = np.tile(normWave,(nOrder,1)).transpose()
    xPowers = x2D**np.arange(nOrder)
    
    return np.dot(xPowers,np.flip(Carr,0))

def extinct(wavel,rad=1.0,n=complex(1.825,-1e-4),logNorm=False,
            npoint=128,pdfThreshold=0.001,s=0.5):
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
        ## Make a linear space from threshold to threshold in the PDF
        lowEval, highEval = invLognorm(s,rad,pdfThreshold)
        
        sizeEvalExtra = np.logspace(np.log(lowEval),np.log(highEval),base=np.e,num=(npoint+1))
        sizeEval = sizeEvalExtra[0:-1] ## extra point is for differentials
        dSize = np.diff(sizeEvalExtra)
        
        weights = lognorm(sizeEval,s,rad) * dSize
        sumWeights = np.sum(weights)
        if (sumWeights < 0.8) | (sumWeights > 1.1):
            print(r'!!!!!!Warning, PDF weights not properly sampling PDF!!!!')
            pdb.set_trace()
        weights = weights / sumWeights
        ## Arrange the array into 2D for multiplication of the grids
        sizeMult = (np.tile(sizeEval/rad,(nwav,1))).transpose()
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
        finalQext = np.dot(weights,qext2D)
    else:
        finalQext = qext
        
    return finalQext

def invLognorm(s,med,pdfThreshold):
    """ 
    Calculates the X values for a Log-normal distribution evaluated at specific PDF values
    Arguments
    ------------------
    s: float
        The sigma (scale value) of the log-normal distribution
    med: float
        The median particle size
    pdfThreshold: float
        The PDF threshold at which to find the x values
    """
    mu = np.log(med)
    z = np.log(s * np.sqrt(2. * np.pi) * pdfThreshold)
    sqrtPart = np.sqrt((2 * s**2 - 2 * mu)**2 - 4. * mu**2 - 8. * s**2 * z)
    lowEval = np.exp(mu - s**2 - 0.5 * sqrtPart)
    highEval = np.exp(mu - s**2 + 0.5 * sqrtPart)
    
    return lowEval, highEval

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
    y = 1. / (s* x * np.sqrt(2.*np.pi)) * np.exp(-0.5*((np.log(x)-mu)/s)**2)
    return y

def xyLogNorm(s,rad,npoint=1024,pdfThresh=0.001,verbose=False):
    """
    Generates x and y coordinates and delta-xs for a log-normal distribution
    """
    lowEval, highEval = invLognorm(s,rad,pdfThresh)
    xExtra = np.logspace(np.log(lowEval),np.log(highEval),num=npoint+1,base=np.e)
    x = xExtra[0:-1]
    dx = np.diff(xExtra)
    y = lognorm(x,s,rad)
    if verbose == True:
        print('Low, High='+str(lognorm(np.array([lowEval,highEval]),s,rad)))
    return x,y,dx

def showLognorm():
    """ Shows example log-normal distributions"""
    rad = 1.0
    #sArray = [0.4,0.5,0.6]
    sArray = [2.5,1.0,0.5,0.25,0.1]
    plt.close('all')
    fig, ax = plt.subplots()
    for oneS in sArray:
        x, y, dx = xyLogNorm(oneS,rad,verbose=True)
        ax.plot(x,y,label='$\sigma$='+str(oneS)+', mu='+"{:.2f}".format(np.log(rad)))
        
        print('Integral='+str(np.sum(y * dx)))
        
    #ax.set_xscale('log')
    ax.set_xlim(-0.1,3)
    ax.legend(loc='best')
    fig.show()

def compareTest():
    """
    Compares the Single Particle and Log-normal distributions
    """
    x = np.linspace(0.2,15,1024)
    rad = 1.0
    nInd=complex(1.825,-1e-4)
    ysingle = extinct(x,rad=rad,logNorm=False,n=nInd)
    plt.loglog(x,ysingle,label='Single Particle')
    npointA = [32,64,128,256]
    for npoint in npointA:
        ymulti = extinct(x,rad=rad,logNorm=True,npoint=npoint,n=nInd)
        plt.plot(x,ymulti,label='Log Normal N='+str(npoint)+',[0.2,5]')
    yWider = extinct(x,rad=rad,logNorm=True,npoint=256,n=nInd)
    plt.plot(x,yWider,label='Log normal N=256,[0.1,10]')
    plt.legend(loc='best')
    plt.show()
    
def getPoly(pord=15):
    """
    Fits a polynomial to a Mie extinction curve
    """
    x = np.linspace(0.2,15,1024)
    rad = 1.0
    nInd = complex(1.67,-0.006)
    y = extinct(x,rad=rad,logNorm=True,npoint=1024,n=nInd)
    polyFit = es_gen.robust_poly(x,y,pord,sigreject=100.)
    plt.loglog(x,y,label='Log-Normal Q$_{ext}$')
    plt.plot(x,np.polyval(polyFit,x),label='Polynomial Fit')
    plt.legend(loc='best')
    
    fitDict = {'Simple n='+str(nInd):{'coefficients':polyFit.tolist()}}
    with open('parameters/mie_params/simplePoly.yaml','w') as outFile:
        yaml.dump(fitDict,outFile,default_flow_style=False)
    plt.show()
    