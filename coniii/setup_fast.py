# 2014-01-25
# To run: python setup_fast.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("fast", ["fast.pyx"],include_dirs=[numpy.get_include()]),
                   Extension("custom_maxent", ["custom_maxent.pyx"],include_dirs=[numpy.get_include()])]
)
