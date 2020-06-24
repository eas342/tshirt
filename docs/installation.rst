Installation
==========================

Currently, :code:`tshirt` should be installed from source using the following steps.
Users should install the :code:`astroconda` package: https://astroconda.readthedocs.io/en/latest/

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


- Activate the astroconda environment
   :code:`conda activate astroconda`
- Navigate to the :code:`tshirt` directory:
   :code:`cd tshirt`
- Pull the latest code from github
   :code:`git pull`
- Reinstall
   :code:`python setup.py install`
- Make sure the new code is used
   .. code-block:: python
   
      from tshirt.pipeline import spec_pipeline
      from importlib import reload
      reload(spec_pipeline)

