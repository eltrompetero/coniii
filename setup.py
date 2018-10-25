# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'pypi_description'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='coniii',
      version='0.3.0',
      description='Convenient Interface to Inverse Ising (ConIII)',
      long_description=long_description,
      url='https://github.com/eltrompetero/coniii',
      author='Edward D. Lee, Bryan C Daniels',
      author_email='edlee@alumni.princeton.edu',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
      ],
      python_requires='>=3.6',
      keywords='inverse Ising maxent maximum entropy inference',
      packages=find_packages(),
      install_requires=['multiprocess==0.70.5',
                        'jupyter>=1',
                        'matplotlib',
                        'scipy',
                        'numpy',
                        'numba>=0.39.0,<1',
                        'dill',
                        'joblib'],
      include_package_data=True,
      package_data={'coniii':['setup_module.py','usage_guide.ipynb']},  # files to include in coniii directory
      py_modules=['coniii.enumerate',
                  'coniii.general_model_rmc',
                  'coniii.ising',
                  'coniii.mc_hist',
                  'coniii.mean_field_ising',
                  'coniii.pseudo_inverse_ising',
                  'coniii.samplers',
                  'coniii.solvers',
                  'coniii.utils']
)
