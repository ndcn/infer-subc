"""
infer_subc base module.

This is the principal module of the infer_subc project.
here you put your main classes and objects.
"""
import numpy as np
import dask.array as da
import numpy as np
import xarray as xr

import scipy
from scipy import ndimage as ndi

# example constant variable
from .constants import *

# from .img_util import (log_transform,
#     inverse_log_transform,
#     unstretch,
#     threshold_li_log,
#     threshold_otsu_log,
#     threshold_multiotsu_log)
