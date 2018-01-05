# CONvenient Interface to Inverse Ising
Authors: Edward D Lee, Bryan C Daniels

Python package for solving maximum entropy problems with a focus on the pairwise maximum entropy
model, also known as the inverse Ising problem.

See "Usage guide.ipynb" for usage documentation and examples.

This project is currently being developed and will be published with an accompanying guide on arXiv.
Any feedback would be greatly appreciated!

## Installation
This package is available on PyPI for Linux and MacOS. It can be installed by running  
>`pip install coniii`

If you have trouble using `pip` and PyPI, then you can always build this from source. Download this
package from GitHub and move the `coniii` folder to wherever you would like to have the module.
Inside `coniii`, you must run
> `python setup_module.py build_ext --inplace`

If you would like to use the `Enumerate` solver for system sizes greater than 9, you must run
exact.py to write those files yourself. This can be run from the install directory.  If you do not
know where the installation directory is, you can find it by opening up an iPython notebook and
running
> `import coniii`  
> `coniii.__path__`

Then, run
>`python exact.py [N]` 

where `[N]` should be replaced by the size of the system.
