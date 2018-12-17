[![PyPI version fury.io](https://badge.fury.io/py/coniii.svg)](https://pypi.python.org/pypi/coniii/) [![PyPI license](https://img.shields.io/pypi/l/coniii.svg)](https://pypi.python.org/pypi/coniii/)

# Convenient Interface to Inverse Ising

ConIII is a Python package for solving maximum entropy problems with a focus on the
pairwise maximum entropy model, also known as the inverse Ising problem. Support for
Python 3.6 and higher only (v0.2.10 works with Python 2.7 but is no longer actively
maintained).

If you use ConIII for your research, please cite the following:
> Lee, Edward D. and Daniels, Bryan C.  Convenient Interface to Inverse Ising (ConIII): A
> Python package for solving maximum entropy models.  arXiv preprint:1801.08216 (2018).

[Documentation](https://eddielee.co/coniii/index.html "Documentation").

## Installation

This package is available on PyPI. It can be installed by running 
```bash
$ pip install coniii
```

If you have trouble using `pip` and PyPI, then you can always build this package from
source.  Download the latest release from GitHub. Make sure that you are running Python
3.6 or higher.  Inside "coniii", you must run
```bash
$ pip install . 
```

Note: Using setuptools in the usual way of `python setup.py install` will not work because
eggs are incompatible with cached jit functions.

If you would like to use the `Enumerate` solver for system sizes greater than 9, you must
run enumerate.py to write those files yourself. This can be run from the install
directory.  If you do not know where the installation directory is, you can find it by
starting a Python terminal and running
```python
>>> import coniii
>>> coniii.__path__
```

Once inside the install directory, you can run in your bash shell
```bash
$ python enumerate.py [N]
```

where `[N]` should be replaced by the size of the system. This will write the equations
for the Ising model in the {0,1} basis. On the other hand,

```bash
$ python enumerate.py [N] 1
```

specifies that the system should be written for the {-1,1} basis.  For more details, see
the `__main__` block at the end of the file enumerate.py.

## Quick guide

A Jupyter notebook with a brief introduction and examples for how to use ConIII is
available in the "ipynb" directory on the GitHub repository:
[https://github.com/eltrompetero/coniii/blob/master/ipynb/usage_guide.ipynb].  The
notebook is installed into your package directory if you used pip, or you can download it
from the above GitHub link.

To use the notebook, install jupyter. 
```bash
$ pip install jupyter
```
or if you are using the Conda package manager
```bash
$ conda install jupyter
```

Then, first copy the notebook file "usage_guide.ipynb" into a directory outside the
"coniii" directory.  Change to this directory and run
```bash
$ jupyter notebook
```

This should open the notebook in your default web browser.

## Dependencies

In order to open the usage guide Jupyter notebook, you must have both Jupyter installed.
To run the examples, you need a number of packages listed in "setup.py". These are all
automatically installed into your Python path when you install ConIII through listed
dependencies.

If you prefer to install the packages yourself, you can use the Python package
[pip](https://pypi.org/project/pip/).  Open a terminal and run
```bash
$ pip install matplotlib 'multiprocess>=0.70.5,<1' matplotlib scipy numpy 'numba>=0.39.0,<1' dill joblib
```

## Troubleshooting

This package is only maintained for Python 3 as of v1.0.2 and has only been tested for
Python 3.6 up to 3.7.1. Check which version of Python you are running in your terminal
with 
```bash
$ python --version
```

ConIII has been tested on the following systems
* Debian 9 (Stretch)
* Mac OS X 10.12 (Sierra)
* Mac OS X 10.13 (High Sierra)

multiprocess module problems: `n_cpus` kwarg can be set to 0 or 1 when the algorithm class
instance is declared.  This will disable the parallel computing functionality.

### Support

Please file an issue on the GitHub if you have any problems or feature requests. Provide a
stack trace or other information that would be helpful in debugging. For example, OS,
system configuration details, and the results of unit tests. These can be run by
navigating to the package directory and running

```bash
$ pytest -q
```

The package directory can be found by running inside python
```python
>>> import coniii
>>> coniii.__path__
```

You may also need to install pytest.
```bash
$ pip install pytest
```

### Updating

When updating to new 1.x.x versions please read the
[RELEASE_NOTES](https://github.com/eltrompetero/coniii/blob/py3/RELEASE_NOTES). There may
be modifications to the interface including parameter names as we make future versions
more user friendly.

[Documentation](https://eddielee.co/coniii/index.html "Documentation").
