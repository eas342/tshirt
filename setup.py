from distutils.core import setup
#from setuptools import setup, find_packages

setup(
    name='tser-tools',
    version='0.1dev',
    packages=['tser_tools','tser_tools.pipeline',
              'tser_tools.pipeline.instrument_specific'],
    license='MIE',
    package_data={'': ['parameters/phot_params/keywords_for_phot_pipeline.csv'],
                  '': ['parameters/spec_params/default_params.yaml'],
                  '': ['parameters/phot_params/keywords_for_phot_pipeline.csv']},
    include_package_data=True,
    long_description=open('README.rst').read(),
)
