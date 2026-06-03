[![tests](https://github.com/eltrompetero/coniii/actions/workflows/test.yml/badge.svg?branch=dev)](https://github.com/eltrompetero/coniii/actions/workflows/test.yml) [![PyPI version fury.io](https://badge.fury.io/py/coniii.svg)](https://pypi.python.org/pypi/coniii/) [![PyPI license](https://img.shields.io/pypi/l/coniii.svg)](https://pypi.python.org/pypi/coniii/)

# Convenient Interface to Inverse Ising

ConIII is a Python package for solving maximum entropy problems with a focus on the
pairwise maximum entropy model, also known as the inverse Ising problem.

If you use ConIII for your research, please consider citing the following:
> Lee, E.D. and Daniels, B.C., 2019. Convenient Interface to Inverse Ising (ConIII): A
> Python 3 Package for Solving Ising-Type Maximum Entropy Models. Journal of Open Research
> Software, 7(1), p.3. DOI: http://doi.org/10.5334/jors.217.

The paper also contains an overview of the modules. For code documentation, see
[coniii.readthedocs.io](https://coniii.readthedocs.io/en/latest/ "Documentation").

## Package layout

| Area | Status | What's there |
| --- | --- | --- |
| `coniii.solvers` | stable | Inverse-Ising solvers: `Enumerate`, `SparseEnumerate`, `MPF`, `MCH`, `Pseudo`, `ClusterExpansion`, `RegularizedMeanField`. |
| `coniii.samplers` | stable | Monte Carlo samplers: `Metropolis`, `WolffIsing`, `ParallelTempering`, `Potts3`, and the `sample_ising` helper. |
| `coniii.models` | stable | Maxent model classes: `Ising`, `Triplet`, `Potts3`. |
| `coniii.utils` | stable | Indexing, correlation, parameter-conversion, and helper-function utilities. |
| `coniii.enumerate`, `coniii.enumerate_potts`, `coniii.ising_eqn` | stable | Exact enumeration and the generated equation files it produces. |
| `coniii.experimental` | work in progress | Unvalidated samplers (`SWIsing`, `HamiltonianMC`, `Heisenberg3DSampler`) and an entropy stub. Not imported by `import coniii`; import explicitly. |
| `coniii.legacy` | deprecated | `mean_field_ising` (still used internally by `ClusterExpansion`/`RegularizedMeanField`) and `pseudo_inverse_ising`. Direct imports warn. |

The top-level names available via `from coniii import *` are listed in `coniii.__all__`.

## Installation

To set up an Anaconda environment called "test" and install from pip, run the following code. The openblas package is only recommended for AMD users.
```bash
$ conda create -n test -c conda-forge python=3.10 numpy scipy numba cython jupyter ipython multiprocess boost==1.74 matplotlib mpmath blas=*=openblas
$ pip install coniii
```
If you have trouble using `pip`, you can build from source. Make sure you are running
Python 3.10 and have boost v1.74.0 installed.
```bash
$ git clone https://github.com/eltrompetero/coniii.git
$ cd coniii
$ pip install .
```
The build probes for Boost and compiles the optional C++ samplers extension if it is
found; without Boost it falls back to the (slower) pure-Python samplers. To build
distributable wheels instead, run `python -m build` (or the convenience wrapper
`./pypi_compile.sh`).

#### Setting up exact solution for systems *N > 9*
If you would like to use the `Enumerate` solver for system sizes greater than 9 spins, you
must run enumerate.py to write those files yourself. This can be run from the install
directory.  If you do not know where the installation directory is, you can find it by
starting a Python terminal and running
```python
>>> import coniii
>>> coniii.__path__
```

You can run the generator from anywhere as a module:
```bash
$ python -m coniii.enumerate [N] 1
```

where `[N]` should be replaced by the size of the system. The trailing `1` specifies the {-1,1} basis. Note that the package uses the {-1,1} basis by default. For more details, see the `__main__` block at the end of `enumerate.py`.

For the {0,1} basis, use
```bash
$ python -m coniii.enumerate [N]
```

## Quick guide with Jupyter notebook

A [Jupyter
notebook](https://github.com/eltrompetero/coniii/blob/py3/ipynb/usage_guide.ipynb) with a
brief introduction and examples for how to use ConIII is available. The
notebook is also installed into your package directory if you used pip.

To use the notebook, install jupyter such as by following the setup instructions above. Then, copy the notebook file "usage_guide.ipynb" into a directory outside the "coniii" directory. Change to this directory and run
```bash
$ jupyter notebook
```

This should open the notebook in your default web browser.

## Troubleshooting

This package is only maintained for Python 3 and has only been tested for Python
3.10. Check which version of Python you are running in your terminal with 
```bash
$ python --version
```

ConIII has been tested on the following systems
* Ubuntu 20.04.5

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
$ conda install -c conda-forge pytest
```

### Updating

When updating, please read the [RELEASE_NOTES](https://github.com/eltrompetero/coniii/blob/py3/RELEASE_NOTES). There may
be modifications to the interface including parameter names as we make future versions
more user friendly. **Note:** v4.0.0 contains breaking changes — see the RELEASE_NOTES
before upgrading.

[Documentation](https://coniii.readthedocs.io/en/latest/ "Documentation").
