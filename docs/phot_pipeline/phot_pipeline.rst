
Usage for Time Series Aperture Photometry
==========================================

Photometry Parameters File
---------------------------
Create a photometry parameters file from the example in the `example_phot_parameters.yaml <parameters/phot_params/example_phot_parameters.yaml>`_.
You will need to specify a list of files, source name and a list of source coordinates in pixels [x,y].
The first source should be the target and the rest will be reference stars.
The ``jdRef`` parameter specifies a reference epoch for time series plots.

Source Aperture Photometry Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the aperture geometry, sizes as well as the box finding size for locating sources. The options for geometry `srcGeometry` are "Circular" and "Rectangular".

Background Aperture Photometry Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specify the background aperture geometry. You can use either ``CircularAnnulus`` or ``Rectangle``. If using ``CircularAnnulus``, the inner and outer radii are set by the ``backStart`` and the ``backEnd`` keywords, while ignoring ``backHeight`` and ``backWidth``. If using ``Rectangular``, the ``backHeight`` and ``backWidth`` keywords are used to define the background aperture size. Regardless of the geometry, the background aperture is always centered relative to the source aperture. The offset between background aperture and the source aperture is set by ``backOffset``, which is a 2 element list in the form of [DX, DY], where DX and DY are the offset in number of pixels.

Fixed Aperture Sizes
~~~~~~~~~~~~~~~~~~~~~~~
For the circular aperture ``apRadius``, ``apHeight`` and ``apWidth`` give the source radius, and the inner and outer radii of the background annulus. For a rectangular aperture, the ``apHeight`` and ``apWidth`` describe the height and width. These units are in pixels.


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

=====================


Running the Time Series Aperture Photometry
--------------------------------------------
Run the following commands in either an iPython session or in a python script:

::

   phot = phot_pipeline.phot(paramFile='parameters/phot_params/example_phot_parameters.yaml')
   phot.showStarChoices()
   phot.showStarChoices(showAps=True)
   phot.do_phot()
   phot.showCustSet()

where ``parameters/phot_params/example_phot_parameters.yaml`` is the name of the ``yaml`` parameters file. 
``phot.showStarChoices`` has a boolean parameter. If ``showAps`` is True, then it will draw the apertures. Sometimes these apertures are too small to be seen easily so ``showAps`` is False will draw circles around the sources.

Re-centering Apertures
~~~~~~~~~~~~~~~~~~~~~~~
Sometimes, the first attempt with centering will fail but for speed's sake, the ``phot.do_phot()`` method, will read the previous centroid data rather than re-fit Gaussians. To redo the centering and overwrite the centroid file, run ``phot2.get_allimg_cen(recenter=True)``.
