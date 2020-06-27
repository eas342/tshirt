Image Reduction
==================================
Image "reduction" turns raw data into processed data.
Typically the following steps are applied:

- Dark/Bias subtraction
- Flat fielding
- Bad Pixel Correction
- Non-linearity correction

``tshirt`` can apply these steps to common CCD data.



How to Use for Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Edit the :doc:`Reduction parameters<reduction_param>`. You can specify any number of directories for reducing data.
Execute ``nohup python prep_images.py &`` to run the reduction in the background.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self
   reduction_param
   