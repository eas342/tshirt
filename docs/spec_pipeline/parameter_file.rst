Spectroscopic Pipeline Parameters
==================================

The spectroscopic pipeline creates a :code:`spec` object from a parameter file.
The example below shows the default parameters.
A library of parameter files is available at https://github.com/eas342/tshirt/tree/master/tshirt/parameters/spec_params

.. literalinclude:: ../../tshirt/parameters/spec_params/default_params.yaml

..   :language: yaml


Notes on Parameters
====================

The following gives some more information on the parameters in the spectroscopic pipeline.

apWidth
~~~~~~~
The full width of the extraction box used for the source. It is going to be centered on :code:`starPositions` with :code:`apWidth`/2 above and :code:`apWidth`/2 below the :code:`starPositions` value. The equivalent IRAF :code:`apall` "lower" and "upper" values would both be :code:`apwidth`/2 each.

mosBacksub
~~~~~~~~~~~

If True, background subtraction is done individually for each source. This only subtracts locally around the dispersion pixels and spatial pixels of the source. The 

bkgRegionsX
~~~~~~~~~~~
The pixels for which you want to do background subtraction. Must be a list of lists. When :code:`mosBacksub` is True, the numbers are the pixels relative to a spectral source. Otherwise, they are absolute pixel dimensions of the image.
If you have a source region that you want to subtract over from 10 to 20 pixels and :code:`mosBacksub` is False, you might set this to:

.. code-block:: python

   bkgRegionsY: ## a list of background regions in the Y direction
      - [0,10]
      - [20,30]


bkgRegionsY
~~~~~~~~~~~~
The same as background regions X.

dispPixels
~~~~~~~~~~~
The absolute pixel region over which to do spectral extractions. When :code:`mosBacksub` is True, the absolute pixels are used for the first source and then the :code:`dispOffsets` parameter shifts it for all other sources.


numSplineKnots
~~~~~~~~~~~~~~
This is a critical parameter in finding the spectroscopic profile. If it is too large, the profile, normalization and variance-weighted fit can be driven to huge numbers. If it is too small, the profile fit will not capture the shape of the spectrum and my chop it off at its peak. You can check the profile fitting by running. The following

.. code-block:: python

    img, head = spec.get_default_im()
    imgSub, bkgModel, subHead = spec.do_backsub(img,head,directions=spec.param['bkgSubDirections'])
    profileList, smooth_img_list = spec.find_profile(imgSub,subHead,showEach=True)
    
This will show you one profile fit at a time with the spline knots shown. You can step through each cross-dispersion pixel by pressing "c". When you are done, press "q" to quick the python debugger (pdb).
 

waveCalMethod
~~~~~~~~~~~~~~
The method to turn the dispersion pixels into wavelengths.

* :code:`None` When it is :code:`None` (:code:`null` in the YAML file), the wavelengths are equal to the pixel in microns (just as a placeholder)

* :code:`NIRCamTSquickPoly` Quick polynomial fit to NIRCam grism time series from before flight measurement (use with caution)

* :code:`wfc3Dispersion` Hubble Space Telescope Wide Field Camera 3 quick wavecal (use with caution)

* :code:`quick_nrs_prism` Simple Polynomial fit to the NIRSpec prism using the jwst pipeline evaluated at Y=16 on 2022-07-15 (use with caution)

* :code:`grismr_poly_dms` A polynomial fit to flight data from program 1076. Should be accurate to within a few angstroms for F322W2. F444W depends on where the target position lands after position adjustments.

saveSpatialProfileStats
~~~~~~~~~~~~~~~~~~~~~~~
If True, save the spatial profile centroid and FWHM for de-trending from the median profile. If False, those are populated with NaN.

profilePix
~~~~~~~~~~
If None, all dispersion pixels (along the wavelength direction) are used in calculating spatial profile statistics. If a 2 element list like :code:`[50,100]`, pixels 50 through 100 in the dispersion/wavelength direction will be used to measure the spatial profile statistics. Note that this only has an effect if :code:`saveSpatialProfileStats` is :code:`True` and this will not affect the extraction in current versions of tshirt. It will ony affect the saved profile statistics.

useSmoothProfileForStats
~~~~~~~~~~~~~~~~~~~~~~~~
Use the smoothed profile for the profile statistics? If True, the statistics will be on the man profile along the dispersion direction. Otherwise, a median of the data is calculated. Note that if :code:`fixedProfile` is True, this will give constant statistics for all images

traceCurvedSpectrum
~~~~~~~~~~~~~~~~~~~~
Trace a curved spectrum? If True, allows a curved aperture instead of a rectangular one

traceOrder
~~~~~~~~~~~
Polynomial order for the trace fitting

traceFitBoxSize
~~~~~~~~~~~~~~~~
The spatial box size used to fit the trace

traceFWHMguess
~~~~~~~~~~~~~~
Guess size for the spatial FWHM when fitting to a Gaussian

traceReference
~~~~~~~~~~~~~~
If you want to use the trace from a previous file, give the path name as a string
