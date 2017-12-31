#! /usr/bin/env/python

# Code for compiling fast Cython module.
# To run: python setup_module.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

def run_setup():
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("fast", ["fast.pyx"],include_dirs=[numpy.get_include()])]
                       #Extension("custom_maxent", ["custom_maxent.pyx"],include_dirs=[numpy.get_include()])]
    )

if __name__=='__main__':
    # Compile fast library.
    run_setup()
    
    # Write Ising equation files and put them into ./ising_eqn directory.
    from exact import main
    if not os.path.exists('ising_eqn'):
        os.makedirs('ising_eqn')
        open('ising_eqn/__init__.py','w').write('')
    for n in xrange(2,10):
        main(n)
        main(n,True)
        os.rename('ising_eqn_%d.py'%n,'ising_eqn/ising_eqn_%d.py'%n)
        os.rename('ising_eqn_%d_sym.py'%n,'ising_eqn/ising_eqn_%d_sym.py'%n)

