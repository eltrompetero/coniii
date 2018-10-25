#! /usr/bin/env/python
# Author : Edward Lee, edlee@alumni.princton.edu
#
# MIT License
# 
# Copyright (c) 2017 Edward D. Lee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    from .exact import main
    if not os.path.exists('ising_eqn'):
        os.makedirs('ising_eqn')
        open('ising_eqn/__init__.py','w').write('')
    for n in range(2,10):
        main(n)
        main(n,True)
        os.rename('ising_eqn_%d.py'%n,'ising_eqn/ising_eqn_%d.py'%n)
        os.rename('ising_eqn_%d_sym.py'%n,'ising_eqn/ising_eqn_%d_sym.py'%n)

