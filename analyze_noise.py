#import phot_pipeline
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from astropy.table import Table

t = Table.read('tser_data/refcor_phot/refcor_phot_S02illum1miAll_GLrun104.fits')


def bin_examine():
    allStd, allBinSize = [], []

    binsTry = [500,100,50,30,10]
    for oneBin in binsTry:
        yBin, xEdges, binNum = binned_statistic(t['Time'],t['Y Corrected'],
                                                statistic='mean',bins=oneBin)
        stdBinned = np.std(yBin)
        binSize = (xEdges[1] - xEdges[0]) * 24. * 60.
        allStd.append(stdBinned)
        allBinSize.append(binSize)

    print("{} bin ({} min) std = {} ppm".format(binsTry[-1],allBinSize[-1],allStd[-1]))
    plt.loglog(allBinSize,allStd)
    plt.show()
    print
