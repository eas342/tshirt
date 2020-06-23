from distutils.core import setup
#from setuptools import setup, find_packages

setup(
    name='tshirt',
    version='0.1dev',
    packages=['tshirt','tshirt.pipeline',
              'tshirt.pipeline.instrument_specific'],
    license='MIE',
    package_data={'': ['parameters/phot_params/keywords_for_phot_pipeline.csv'],
                  '': ['parameters/spec_params/default_params.yaml'],
                  '': ['parameters/phot_params/keywords_for_phot_pipeline.csv']},
    include_package_data=True,
    long_description=open('README.rst').read(),
)
