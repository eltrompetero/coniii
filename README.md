# Convenient Interface to Inverse Ising
Edward D Lee, Bryan C Daniels

Python package for solving maximum entropy problems with a focus on the pairwise maximum entropy
model, also known as the inverse Ising problem. Support for Python 3.6 only (v0.2.5 works with Python
2.7 but is no longer actively maintained).

See "ipynb/usage_guide.ipynb" for usage documentation and examples.

There is an accompanying guide on arXiv: [https://arxiv.org/abs/1801.08216]. Please cite this if you
use ConIII for your research! Citations fuel further open source work of this kind.

## Installation
This package is available on PyPI for Linux and MacOS. It can be installed by running  
>`pip install coniii`

If you have trouble using `pip` and PyPI, then you can always build this package from source.  Download this
package from GitHub and move the `coniii` folder to wherever you would like to have the module.
Inside `coniii`, you must run
> `python setup_module.py build_ext --inplace`

If you would like to use the `Enumerate` solver for system sizes greater than 9, you must run
exact.py to write those files yourself. This can be run from the install directory.  If you do not
know where the installation directory is, you can find it by opening up an iPython notebook and
running
> `import coniii`  `coniii.__path__`

Once inside the install directory, you can run in your bash shell
>`python exact.py [N]` 

where `[N]` should be replaced by the size of the system.

## Usage examples
The Jupyter notebook with the examples for how to use CONIII is available in the ipynb directory on
the Github: 
[https://github.com/bcdaniels/coniii/blob/master/ipynb/usage_guide.ipynb]

In order to run the usage guide Jupyter notebook, you must have both jupyter and matplotlib
installed. These can be fetched with
>`pip install jupyter matplotlib`

However, we strongly recommend that you use the Anaconda package manager
[https://www.anaconda.com/download/] in which case you can install the packages by running
>`conda install jupyter matplotlib`

The notebook is also installed into your environment directory if you used pip. If you do not know how to
navigate to your Python site-packages directory, I would recommend that you just download it from
the Github link provided above.

Make sure you copy the notebook into a directory outside the CONIII directory. Once in the directory
into which you've installed it, you can run it with
> jupyter notebook

Make sure that you are running Python 3.6. If you run
> python

The first few lines specifying the version of Python that you are using must be 3.6. This package is
not supported for Python 2.

## Troubleshooting
Some users may encounter difficulties with the multiprocess module in which case the `n_cpus` kwarg should be set to 0 when the algorithm class instance is declared.
