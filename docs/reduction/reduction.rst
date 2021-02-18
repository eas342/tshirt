Image Reduction Pipeline
==================================
Image "reduction" turns raw data into processed data.
Typically the following steps are applied:

- Dark/Bias subtraction
- Flat fielding
- Bad Pixel Correction
- Non-linearity correction

:code:`tshirt` can apply these steps to common CCD data.



How to Use for Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Edit the :doc:`Reduction parameters<reduction_param>`. You can specify any number of directories for reducing data.

Start by reading in the parameters and making calibration files.

.. code-block:: python

   from tshirt.pipeline import prep_images
   pipeObj = prep_images.prep('parameters/path_to_file.yaml')
   pipeObj.makeMasterCals()


After the calibration files look good, apply them to the science data

.. code-block:: python

   pipeObj.procSciFiles()




.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self
   reduction_param
   