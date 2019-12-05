import photutils
from astropy.io import fits, ascii
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
from astropy.time import Time
import astropy.units as u
import pdb
from copy import deepcopy
import yaml
import warnings
from scipy.stats import binned_statistic
from astropy.table import Table
import multiprocessing
from multiprocessing import Pool


maxCPUs = multiprocessing.cpu_count() // 3

