#from distutils.core import setup
from setuptools import setup#, find_packages

with open("README.md", "r") as fh:
    ## skip the HTML, which doesn't work on PyPI
    long_description = "".join(fh.readlines()[4:])

setup(
    name='tshirt',
    version='0.1dev5',
    author='Everett Schlawin, Kayli Glidic',
    packages=['tshirt','tshirt.pipeline',
              'tshirt.pipeline.instrument_specific'],
    url="https://github.com/eas342/tshirt",
    description="A package to analyze time series data, especially for exoplanets",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
