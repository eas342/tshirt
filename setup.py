from distutils.core import setup
#from setuptools import setup, find_packages

setup(
    name='tshirt',
    version='0.1dev',
    packages=['tshirt','tshirt.pipeline',
              'tshirt.pipeline.instrument_specific'],
    license='MIE',
    package_data={'tshirt': ['parameters/phot_params/keywords_for_phot_pipeline.csv'],
                  'tshirt': ['parameters/spec_params/default_params.yaml']},
    long_description=open('README.rst').read(),
)
