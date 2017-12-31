# CONvenient Interface to Inverse Ising
Authors: Edward D Lee, Bryan C Daniels

Python package for solving maximum entropy problems with a focus on the pairwise maximum entropy
model, also known as the inverse Ising problem.

See "Usage guide.ipynb" for usage documentation and examples.

This project is currently being developed and will be published with an accompanying guide on arXiv.
Any feedback would be greatly appreciated!

## Installation
This package is available on PyPI. It can be installed by running  
>`pip install coniii`

After installation, some of the remaining Cython libraries must be compiled and the Ising equation
files need to be written for use of the `Enumerate` solver.

The Cython library can compiled by going to the installation directory, opening the coniii
directory, and running  
>`python setup_module.py build_ext --inplace`

If you do not know where the installation directory is, you can find it by opening up an iPython
notebook and running
> `import coniii`  
> `coniii.__path__`

The `Enumerate` solver needs the full system of nonlinear equations to be written out and solves
those numerically. These equations must be written to file first. This can be done by going to the
installation directory (usually found in the home directory under
"anaconda/lib/python2.7/site-packages/coniii/"), opening the coniii directory and running  
>`mkdir ising_eqn
>`python exact.py [N]` 

where `[N]` should be replaced by the size of the system.
