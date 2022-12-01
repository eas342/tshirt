#from distutils.core import setup
from setuptools import setup#, find_packages

with open("README.md", "r") as fh:
    ## skip the HTML, which doesn't work on PyPI
    long_description = "".join(fh.readlines()[4:])

setup(
    name='tshirt',
    version='0.2',
    author='Everett Schlawin, Kayli Glidic',
    packages=['tshirt','tshirt.pipeline',
              'tshirt.pipeline.instrument_specific'],
    url="https://github.com/eas342/tshirt",
    description="A package to analyze time series data, especially for exoplanets",
    include_package_data=True,
    install_requires=[
        "numpy>=1.15",
        "scipy>=1.1.0",
        "astropy>=2.0",
        "tqdm>=4.46.0",
        "photutils>=0.4.1",
        "bokeh>=1.4.0",
        "pytest",
        "celerite2"
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
