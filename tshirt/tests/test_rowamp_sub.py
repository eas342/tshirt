import unittest
from tshirt.pipeline import phot_pipeline, spec_pipeline
from astropy.io import fits,ascii
from pkg_resources import resource_filename
from copy import deepcopy
from tshirt.pipeline import sim_data
import os
import numpy as np

