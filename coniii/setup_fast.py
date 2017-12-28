#! python

# Code for compiling fast Cython module.
# To run: python setup_fast.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

def run_setup():
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("fast", ["fast.pyx"],include_dirs=[numpy.get_include()])]
                       #Extension("custom_maxent", ["custom_maxent.pyx"],include_dirs=[numpy.get_include()])]
    )

if __name__=='__main__':
    run_setup()

