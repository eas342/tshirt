Time Series Tools README
==========================================
This repository contains code for analyzing and reducing time series data.
Example usage includes:

 - Using ``emcee`` to fit light curves with sinusoidal models (``fit_tser_emcee.py``).
 - A photometric pipeline to reduce photometric data: flat field, bias subtract, gain correct, etc. (``prep_images.py``). Tested to work with merged CCD images from the Mont4K imager on the Kuiper-61 inch telescope on Mt Bigelow, AZ.
 - Extract photometry from reduced images. (``phot_pipeline.py``)

Installation
==========================================
 - Clone this git repository and ``pip install`` any dependencies.

Dependencies
----------------------------------
 - ``astropy``, ``numpy``
 - ``ccdphot`` - only needed for data reduction of photometric data
 - ``emcee`` - only needed for time series analysis and model fitting
 - ``miescatter`` - only needed for fitting Mie extinction to spectra. Note: I had trouble with ``pip install miescatter`` where it was looking for an old gcc5 it couldn't find and had to download the source from pyPI and run ``python setup.py install``
 - ``photutils`` - needed for photometric extraction on images

Usage for Data Reduction
==========================================
Edit `parameters/reduction_parameters.yaml <parameters/reduction_parameters.yaml>`_. You can specify any number of directories for reducing data.
Execute ``nohup python prep_images.py &`` to run the reduction in the background.

Usage for Time Series Aperture Photometry
==========================================

Photometry Parameters File
---------------------------
Create a photometry parameters file from the example in ``parameters/phot_parameters.yaml``.
You will need to specify a list of files, source name and a list of source coordinates in pixels [x,y].
The first source should be the target and the rest will be reference stars.
Specify the aperture geometry, sizes as well as the box finding size for locating sources.
The ``jdRef`` parameter specifies a reference epoch for time series plots.
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


Running the Time Series Aperture Photometry
--------------------------------------------
Run the following commands in either an iPython session or in a python script:

::

   phot = phot_pipeline.phot(paramFile='parameters/aug2016_corot1_parameters.yaml')
   phot.do_phot()
   phot.showCustSet()

where ``parameters/aug2016_corot1_parameters.yaml`` is the name of the ``yaml`` parameters file.

Usage for Time Series Model Fitting
====================================