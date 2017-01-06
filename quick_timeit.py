import time
import fit_tser_emcee
import numpy as np

def time_eval1():
    """ Times the evaluation of a Fourier series model"""
    fSmodel = fit_tser_emcee.fSeries()
    x = np.linspace(0,5,1024)
    p = np.array([4.1,0.,0.995,1.5,0.4,0.4,1.38,0.4,0.1])
    tArr = []
    for i in range(30):
        t1 = time.time()
        y = fSmodel.evaluate(x,p)
        t2 = time.time()
        tArr.append(t2-t1)
    avg = np.mean(tArr)
    print('Average time= '+str(avg * 1000.)+' ms')
    
def time
    ## Took 0.0989 ms for 