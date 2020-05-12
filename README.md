[![PyPI version fury.io](https://badge.fury.io/py/coniii.svg)](https://pypi.python.org/pypi/coniii/) [![PyPI license](https://img.shields.io/pypi/l/coniii.svg)](https://pypi.python.org/pypi/coniii/)

# Convenient Interface to Inverse Ising

**ConIII is now on version 2. Major interface updates may break code compatibility from
version 1.  See [release
notes](https://github.com/eltrompetero/coniii/blob/py3/RELEASE_NOTES "release notes").**

ConIII is a Python package for solving maximum entropy problems with a focus on the
pairwise maximum entropy model, also known as the inverse Ising problem. Support for
Python 3.7.5 and higher.

If you use ConIII for your research, please consider citing the following:
> Lee, E.D. and Daniels, B.C., 2019. Convenient Interface to Inverse Ising (ConIII): A
> Python 3 Package for Solving Ising-Type Maximum Entropy Models. Journal of Open Research
> Software, 7(1), p.3. DOI: http://doi.org/10.5334/jors.217.

The paper also contains an overview of the modules. For code documentation, see
[here](https://eddielee.co/coniii/index.html "Documentation").

## Installation

This package is available on PyPI. It can be installed by running 
```bash
$ pip install coniii
```
We highly recommend the use of virtual environments as supported through Anaconda to
manage this package and associated ones that need to be installed.

If you have trouble using `pip`, then you can always build this package from
source.  Download the latest release from GitHub. Make sure that you are running Python
3.7.5 or higher.  Inside the top directory "coniii", you must run 
```bash
$ ./pypi_compile.sh
$ pip install dist/*.whl
```
(Note: Using setuptools in the usual way of `python
setup.py install` will not work because eggs are incompatible with cached jit functions
generated using numba.)

#### Speeding up Metropolis sampler
We provide a much faster version of the Metropolis sampler from the Ising model using a
C++ Boost library implementation. In order to compile this code, you must have Boost v1.72
installed and compiled for Python 3.7. The dynamic libraries libboost_python and
libboost_numpy must also be on your environment path.

[For Mac users: If you do not have the Boost library installed, you can install it using
Homebrew.
```bash
$ brew install boost-python3
```
This will symlink the necessary Boost libraries into `/usr/local/lib`.]

To compile the code, download the source code, and run the compilation script.
```bash
$ git clone https://github.com/eltrompetero/coniii.git
$ cd coniii
$ ./pypi_compile.sh
$ pip install dist/*.whl --force-reinstall --no-deps
```

If the Boost extension refuses to compile, you may need to explicitly specify the path to
your dynamic libraries. For example, I have installed my Boost libraries on
`/usr/local/lib/boost_1_72_0/stage/lib`, so in `setup.py` I edit the `DEFAULT_LIBRARY_DR`
variable to include this path. Alternatively, you can modify the environment path
variables LIBRARY_PATH and LD_LIBRARY_PATH. For a few more hints, see DEVREADME or open an
issue!

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

## Dependencies

In order to open the usage guide Jupyter notebook, you must have both Jupyter installed.
To run the examples, you need a number of packages listed in "setup.py". These are all
automatically installed into your Python path when you install ConIII through listed
dependencies.

If you prefer to install the packages yourself, you can use the Python package
[pip](https://pypi.org/project/pip/).  Open a terminal and run
```bash
$ pip install matplotlib 'multiprocess>=0.70.5,<1' matplotlib scipy 'numpy>=1.15.4,<2' 'numba>=0.43.0,<1' 'mpmath>=1.1.0' dill joblib
```

## Troubleshooting

This package is only maintained for Python 3 as of v1.0.2 and has only been tested for
Python 3.7.5. Check which version of Python you are running in your terminal
with 
```bash
$ python --version
```

ConIII has been tested on the following systems
* Ubuntu 18.04.1
* Mac OS X 10.15 (Catalina)

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
