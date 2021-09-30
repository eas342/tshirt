# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The module contains tools for centroiding sources using Gaussians.
"""

import warnings

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const1D, Const2D, Gaussian1D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

__all__ = ['centroid_2dg']


def centroid_2dg_w_sigmas(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.
    
    ES modified this to spit out the best-fit gaussians
    
    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """
    from photutils.morphology import data_properties  # prevent circular imports

    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'infs) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Gaussian plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.

    mask = data.mask
    data.fill_value = 0.
    data = data.filled()

    # Subtract the minimum of the data as a rough background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties. Moments from negative data
    # values can yield undefined Gaussian parameters, e.g., x/y_stddev.
    props = data_properties(data - np.min(data), mask=mask)
    
    if hasattr(props,'semimajor_sigma'):
        x_stddev = props.semimajor_sigma.value
    else:
        x_stddev = props.semimajor_axis_sigma.value
    
    if hasattr(props,'semiminor_sigma'):
        y_stddev = props.semiminor_sigma.value
    else:
        y_stddev = props.semiminor_axis_sigma.value
    
    constant_init = 0.  # subtracted data minimum above
    g_init = (Const2D(constant_init)
              + Gaussian2D(amplitude=np.ptp(data),
                           x_mean=props.xcentroid,
                           y_mean=props.ycentroid,
                           x_stddev=x_stddev,
                           y_stddev=y_stddev,
                           theta=props.orientation.value))
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    return np.array([gfit.x_mean_1.value, gfit.y_mean_1.value,
                     gfit.x_stddev_1.value,gfit.y_stddev_1.value])
    
