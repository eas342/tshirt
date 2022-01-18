Welcome to tshirt's documentation!
==================================

.. image:: images/t_shirt_logo.jpg
   :width: 200px
   :alt: T shirt logo
   :align: center

The Time Series Helper & Integration Reduction Tool :code:`tshirt` is a general-purpose tool for time series science.
Its main application is transiting exoplanet science.
:code:`tshirt` can:

- Reduce raw data: flat field, bias subtract, gain correct, etc. This has been demonstrated to work with merged CCD images from Mont4K imager on the Kuiper-61 inch on Mt Bigelow, AZ.
- Extract Photometry
- Extract Spectroscopy

The raw code is accessible at https://github.com/eas342/tshirt

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   reduction/reduction
   phot_pipeline/phot_pipeline
   spec_pipeline/spec_pipeline
   specific_modules/specific_modules.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Reference
   
   modules

Indices and tables
---------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
