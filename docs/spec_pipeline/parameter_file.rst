Spectroscopic Pipeline Parameters
==================================

The spectroscopic pipeline creates a :code:`spec` object from a parameter file.
The example below shows the default parameters.
A library of parameter files is available at https://github.com/eas342/tshirt/tree/master/tshirt/parameters/spec_params

.. literalinclude:: ../../tshirt/parameters/spec_params/default_params.yaml

..   :language: yaml


Notes on Parameters
====================

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
 

