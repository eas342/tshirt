Spectroscopic Pipeline
======================

:code:`tshirt` has a pipeline for spectroscopic data that can:

- Do background subtraction of the images
- Find the 2D profile of images
- Extract the spectrum for all images
- Gather together a dynamic spectrum
- Plot and save wavelength-binned data



Batch Processing
----------------
A batch object can iterate over any spec object.

.. code-block:: python

   bspec = spec_pipeline.batch_spec(batchFile='corot1_batch_file.yaml')
   bspec.batch_run('plot_wavebin_series', nbins=1, interactive=True) 
   
This could be done on any method that spec can.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   test_spec_pipeline.ipynb
   parameter_file
   

