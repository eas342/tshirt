Installation
==========================

Currently, :code:`tshirt` should be installed from source using the following steps.


It is recommended that users should install the :code:`astroconda` package: https://astroconda.readthedocs.io/en/latest/
Then proceed to the steps below.

- Activate the astroconda environment
   :code:`conda activate astroconda`
- Navigate to the directory where you want to install :code:`tshirt`:
   :code:`cd Code`
- Clone the repository
   :code:`git clone https://github.com/eas342/tshirt`
- Navigate to the :code:`tshirt` directory:
   :code:`cd tshirt`
- Reinstall
   :code:`python setup.py install`

Upgrading :code:`tshirt`
~~~~~~~~~~~~~~~~~~~~~~~~~~~


- Activate the astroconda environment in bash
   :code:`conda activate astroconda`
- Navigate to the :code:`tshirt` directory:
   :code:`cd tshirt`
- Pull the latest code from github
   :code:`git pull`
- Reinstall
   :code:`python setup.py install`
- Make sure the new code is used
   Either restart your iPython session, restart your Jupyter notebook kernel or run the commands below to update a specific module
   
   .. code-block:: python   
   
      from tshirt.pipeline import spec_pipeline
      from importlib import reload
      reload(spec_pipeline)
      spec = spec_pipeline.spec('path_to_paramfile.yaml')
      
It is important to **re-create** the spec object after the reload, as in the above example.
      
Dependencies
~~~~~~~~~~~~~~~~~~~~
Another option is to manually install dependencies (beta)

- ``astropy``, ``numpy``
- ``photutils`` - needed for photometric extraction on images
- ``ccdphot`` - only needed for flat fielding, dark subtraction etc. for
- ``emcee`` - only needed for time series analysis and model fitting
- ``miescatter`` - only needed for fitting Mie extinction to spectra. Note: I had trouble with ``pip install miescatter`` where it was looking for an old gcc5 it couldn't find and had to download the source from pyPI and run ``python setup.py install``

