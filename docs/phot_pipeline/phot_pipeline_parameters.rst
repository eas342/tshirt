Photometry Parameters File
---------------------------
Create a photometry parameters file like the example on the :doc:`Example Photometric Parameters Page <example_phot_param_file>`.
You will need to specify a list of files, source name and a list of source coordinates in pixels [x,y].
The first source should be the target and the rest will be reference stars.
The ``jdRef`` parameter specifies a reference epoch for time series plots.

Source Aperture Photometry Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the aperture geometry, aperture sizes, and the box finding size for locating sources. The options for geometry `srcGeometry` are "Circular", "Rectangular", and "CircularAnnulus". If using ``CircularAnnulus``, the inner and outer radii are set by the ``srcStart`` and the ``srcEnd`` keywords while ignoring ``apRadius``,  ``apHeight`` and ``apWidth``. If using ``Rectangular``, the ``apHeight`` and ``apWidth`` keywords are used to define the source aperture size while ignoring ``apRadius``,  ``srcStart`` and ``srcEnd``. 

Background Aperture Photometry Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specify the background aperture geometry (:code:`bkgGeometry`). You can use either ``CircularAnnulus`` or ``Rectangle``. If using ``CircularAnnulus``, the inner and outer radii are set by the ``backStart`` and the ``backEnd`` keywords, while ignoring ``backHeight`` and ``backWidth``. If using ``Rectangular``, the ``backHeight`` and ``backWidth`` keywords are used to define the background aperture size. Regardless of the geometry, the background aperture is always centered relative to the source aperture. The offset between background aperture and the source aperture is set by ``backOffset``, which is a 2 element list in the form of [DX, DY], where DX and DY are the offset in number of pixels.


Background Subtraction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specify the background method (:code:`bkgMethod`). The options are
   - "mean" (default). This calculates the mean background value per pixel and subtracts this from all source pixels
   - "median". This calculates the median background value value per pixel and subtracts this from all source pixels
   - "robust mean". This calculates the robust mean background value value per pixel and subtracts this from all source pixels.
   - "colrow". This calculates a column-by-colum and/or row-by-row fit to the background. The parameters are similar to the :doc:`Spec Background Parameters </spec_pipeline/parameter_file>`. Specify the :code:`bkgOrderX`, :code:`bkgOrderY` for the polynomial orders of the fits. For example :code:`bkgOrderX: 1` for a linear fit. The order and which directions to be specified are in the :code:`backsub_directions` parameter. To do :code:`backsub_directions: ['Y','X']` would do the Y direction (column-by-column) first and then the X direction (row-by-row). :code:`backsub_directions: ['X']` would only do row-by-row subtraction.
   - "rowAmp". This is a somewhat JWST-specific code that will do row-by-row subtraction of the whole array (masking out sources) but treats each column individually. The sources are masked by making all pixels with a circle with radius :code:`backStart` from the source Nan and then using :code:`numpy.nanmedian()` to calculate the median of each row within a given amplifier. Mileage may vary if sources extend over an entire amplifier (512 pixels for JWST Stripe mode, also called 4 output amplifier mode).


Fixed Aperture Sizes
~~~~~~~~~~~~~~~~~~~~~~~
For the circular aperture, ``apRadius`` gives the source radius, and ``backStart`` and ``backEnd`` give the inner and outer radii of the background annulus. For a rectangular aperture, the ``apHeight`` and ``apWidth`` describe the height and width of the source, and ``backHeight`` and ``backWidth`` describe the height and width of the background. These units are in pixels.

Scaled Aperture Sizes
~~~~~~~~~~~~~~~~~~~~~~
The apertures can be fixed for all images or be scaled with the FWHM using either ``scaleAperture: True`` or ``scaleAperture: False``. If true, specify the scaling factor. The source aperture will be the FWHM multiplied by the scaling factor 

.. math::

   r_src = FWHM * apScale.

The background start will be calculated as 

.. math::

   r_in = backStart - apRadius + r_src.
   
The background end will be calculated as

.. math::

   r_out = backEnd - backStart + r_in

where the ``apScale``, ``backStart``, ``apRadius`` and ``backEnd`` keywords are specified in the parameter file.
You can also specify an ``apRange`` parameter which sets the minimum and maximum allowed FWHM. This adds some robustness in the case the FWHM found is wacky - for example if clouds go over.


Timing Method
~~~~~~~~~~~~~~~~~~~~~~
The ``phot_pipeline`` will automatically find the JD time from the ``DATE-OBS`` and ``TIME-OBS`` keywords. However, if using JWST data, all the integrations are packed into a singel fits file with one ``DATE-OBS`` and ``TIME-OBS``. In this case, the data must be split into individual integrations, which are assigned an ``ON_INT`` keyword. If ``timineMethod`` is set to ``JWSTint``, then ``phot_pipeline`` will use the calculate integration times using ``TFRAME`` and ``INTTIME`` in the header.

The parameter :code:`dateKeyword` will tell :code:`tshirt` to look for a specific FITS keyword for the date. Otherwise it searches common ones like :code:`DATE-OBS` or :code:`DATE_OBS`.
