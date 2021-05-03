[![PyPI version fury.io](https://badge.fury.io/py/coniii.svg)](https://pypi.python.org/pypi/coniii/) [![PyPI license](https://img.shields.io/pypi/l/coniii.svg)](https://pypi.python.org/pypi/coniii/)

# Convenient Interface to Inverse Ising

ConIII is a Python package for solving maximum entropy problems with a focus on the
pairwise maximum entropy model, also known as the inverse Ising problem. Support for
Python 3.8.3 and higher.

If you use ConIII for your research, please consider citing the following:
> Lee, E.D. and Daniels, B.C., 2019. Convenient Interface to Inverse Ising (ConIII): A
> Python 3 Package for Solving Ising-Type Maximum Entropy Models. Journal of Open Research
> Software, 7(1), p.3. DOI: http://doi.org/10.5334/jors.217.

The paper also contains an overview of the modules. For code documentation, see
[here](https://eddielee.co/coniii/index.html "Documentation").

## Installation

This package is available on PyPI. It can be installed by first installing the needed
Boost C++ libraries and using pip.
```bash
$ conda install -c conda-forge boost==1.74
$ pip install coniii
```
If you have trouble using `pip`, then you can always build this package from
source. The following code will down download the latest release from GitHub and install
the package. Make sure that you are running Python 3.8.3 or higher and have boost v1.74.0
installed.
```bash
$ git clone https://github.com/eltrompetero/coniii.git
$ cd coniii
$ ./pypi_compile.sh
$ pip install dist/*.whl
```
(Note: Using setuptools in the usual way of `python
setup.py install` will not work because eggs are incompatible with cached jit functions
generated using numba.)

#### Setting up exact solution for systems *N > 9*
If you would like to use the `Enumerate` solver for system sizes greater than 9 spins, you
must run enumerate.py to write those files yourself. This can be run from the install
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

specifies that the system should be written for the {-1,1} basis. Note that the package
uses the {-1,1} basis by default. For more details, see the `__main__` block at the end of
the file enumerate.py.

## Quick guide (with Jupyter notebook)

A [Jupyter
notebook](https://github.com/eltrompetero/coniii/blob/py3/ipynb/usage_guide.ipynb) with a
brief introduction and examples for how to use ConIII is available. An HTML version is
[here](https://github.com/eltrompetero/coniii/blob/py3/ipynb/usage_guide.html). The
notebook is installed into your package directory if you used pip.

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

## Quick guide (console)

First, install iPython for a console-based interpreter and start it.
```bash
$ pip install ipython
```
or if you are using the Conda package manager
```bash
$ conda install ipython
```

Then, first copy the notebook file
["usage_guide.py"](https://github.com/eltrompetero/coniii/blob/py3/ipynb/usage_guide.py)
into a directory outside the "coniii" directory.  Change to this directory and run
```bash
$ ipython
```

Once inside the iPython interpreter, run
```python
>>> %run usage_guide.py
```
This will run all the examples sequentially, so you may want to comment out unwanted lines.

## Troubleshooting

This package is only maintained for Python 3 and has only been tested for Python
3.8.3. Check which version of Python you are running in your terminal with 
```bash
$ python --version
```

ConIII has been tested on the following systems
* Ubuntu 18.04
* Ubuntu 20.04.1

Trouble compiling the Boost extension manually? Check if your Boost library is
included in your path. If it is not, then you can add an include directory entry
into the `EXTRA_COMPILE_ARGS` variable in "setup.py" before compiling.


### Support

Please file an issue on the GitHub if you have any problems or feature requests. Provide a
stack trace or other information that would be helpful in debugging. For example, OS,
system configuration details, and the results of unit tests. Unit tests can be run by
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

When updating, please read the [RELEASE_NOTES](https://github.com/eltrompetero/coniii/blob/py3/RELEASE_NOTES). There may
be modifications to the interface including parameter names as we make future versions
more user friendly.

[Documentation](https://eddielee.co/coniii/index.html "Documentation").
