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
