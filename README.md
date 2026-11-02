<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/eltrompetero/coniii/py3/docs/_static/ConIII-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/eltrompetero/coniii/py3/docs/_static/ConIII-logo.svg" alt="ConIII" width="300">
  </picture>
</p>

[![tests](https://github.com/eltrompetero/coniii/actions/workflows/test.yml/badge.svg?branch=dev)](https://github.com/eltrompetero/coniii/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/coniii.svg)](https://pypi.org/project/coniii/)
[![docs](https://readthedocs.org/projects/coniii/badge/?version=latest)](https://coniii.readthedocs.io/en/latest/)
[![license](https://img.shields.io/pypi/l/coniii.svg)](https://github.com/eltrompetero/coniii/blob/py3/LICENSE.txt)

# ConIII: Convenient Interface to Inverse Ising

ConIII is a Python package for solving maximum entropy problems, with a focus on the
pairwise maximum entropy model, also known as the inverse Ising problem. Given samples of
binary variables, it finds the fields h<sub>i</sub> and couplings J<sub>ij</sub> of the
Ising model whose means and pairwise correlations match those of the data. Several
standard solvers share one interface, so you can switch between them or compare them on
the same data.

If you use ConIII in your research, please [cite the paper](#citing-coniii).

## Installation

```bash
$ pip install coniii
```

ConIII requires Python 3.9 or newer and runs on Linux, macOS and Windows. It is not on
conda-forge; in a conda environment, install it with pip.

## Quick start

Data go in a matrix with one row per observation and one column per variable, with values
in {-1, 1}. Convert 0/1 data with `X = 2*X - 1`.

This example samples from an Ising model with known parameters, then solves for them with
pseudolikelihood:

```python
import numpy as np
from coniii import Pseudo, vec2mat
from coniii.samplers import sample_ising

# Draw 10,000 samples of 10 spins from an Ising model with known parameters.
n = 10
rng = np.random.RandomState(0)
true_params = rng.normal(scale=0.2, size=n + n*(n-1)//2)  # fields h_i, then couplings J_ij (i < j)
X = sample_ising(true_params, 10_000, seed=0, parallel=False)  # shape (10000, 10), values in {-1, 1}

# Find the fields and couplings that reproduce the data's means and pairwise correlations.
solver = Pseudo(X)
params = solver.solve()
h, J = vec2mat(params, separate_fields=True)  # h has shape (10,), J has shape (10, 10)

print(np.abs(params - true_params).max())  # largest error in any parameter
```

Every solver follows the same pattern: create it with the data, then call `solve()`. Some
solvers take extra arguments, such as the accuracy threshold for `ClusterExpansion`. The
solved parameters are also stored in `solver.multipliers`.

## Solvers

| Solver | Method | Notes |
| --- | --- | --- |
| `Enumerate` | Exact enumeration of the partition function | Exact, but cost grows as 2<sup>N</sup>, so only for small systems. Equations for N ≤ 9 are included; see [below](#exact-solutions-for-n--9) for larger N. |
| `SparseEnumerate` | Exact enumeration with a subset of the parameters | Parameters you leave out are fixed at zero. |
| `Pseudo` | Pseudolikelihood (Aurell & Ekeberg, PRL 2012) | Fast and scales to large systems. Pairwise Ising model only. |
| `MPF` | Minimum probability flow (Sohl-Dickstein, Battaglino & DeWeese, PRL 2011) | Fast and approximate. Extends to other constraints through `calc_observables`. |
| `ClusterExpansion` | Adaptive cluster expansion (Barton & Cocco, J. Stat. Mech. 2013) | A threshold sets the accuracy; lower is more accurate and slower. Pairwise Ising model only. |
| `MCH` | Monte Carlo histogram (Broderick et al., arXiv 2007) | Works for general maximum entropy problems and can match the correlations as closely as you want. Slowest, since it samples by Monte Carlo, and needs its learning rule tuned. |
| `RegularizedMeanField` | Regularized mean field (Daniels, Krakauer & Flack, Nat. Commun. 2017) | Mean-field solution with regularized couplings; the regularization strength is chosen by Monte Carlo sampling. Pairwise Ising model only. |

The [usage guide notebook](https://github.com/eltrompetero/coniii/blob/py3/ipynb/usage_guide.ipynb)
runs each solver on the same example and compares the results. It also shows a model
with triplet interactions. The notebook is installed with the package; to find it, run

```bash
$ python -c "import coniii, os; print(os.path.join(os.path.dirname(coniii.__file__), 'usage_guide.ipynb'))"
```

Copy it to a directory outside the package before opening it with Jupyter.

The API reference is at [coniii.readthedocs.io](https://coniii.readthedocs.io/en/latest/),
and the [paper](https://doi.org/10.5334/jors.217) describes the methods and the package
design.

### Exact solutions for N > 9

`Enumerate` needs a file of equations for each system size. Files for N ≤ 9 are
included. For a larger system, generate the file by running the `enumerate` module (from
any directory):

```bash
$ python -m coniii.enumerate [N] 1
```

where `[N]` is the system size. The trailing `1` selects the {-1, 1} basis (the package
default); omit it for the {0, 1} basis. See the `__main__` block at the end of
`enumerate.py` for more options.

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

## Optional C++ samplers

The `Metropolis` and `Potts3` samplers have an optional C++ extension built with
Boost.Python. The wheel on PyPI does not include it. To build it, install from source in
an environment that has the Boost headers and the Boost.Python and Boost.NumPy libraries
for your Python version:

```bash
$ git clone https://github.com/eltrompetero/coniii.git
$ cd coniii
$ pip install .
```

The build looks for Boost in the active conda environment, `/usr/include` and
`/usr/local/include`. If it doesn't find Boost, it installs without the extension and the
pure-Python samplers are used. To check whether the extension was built, run

```bash
$ python -c "from coniii import samplers; print(samplers.IMPORTED_SAMPLERS_EXT)"
```

If your Boost is installed somewhere else, add its include directory to `include_dirs` in
the `_detect_boost()` function in `setup.py`.

**Known issue ([#46](https://github.com/eltrompetero/coniii/issues/46)):** on Linux,
building against the system's Boost packages (e.g. Ubuntu's `libboost-python-dev`) links
the extension to a Boost.NumPy library compiled for NumPy 1.x, and `import coniii` then
crashes with a segmentation fault under NumPy 2. Install `numpy<2` in that environment.

The C++ samplers are being replaced by numba versions that need no compiler and are
included in the pip package
([#41](https://github.com/eltrompetero/coniii/issues/41)), which also removes this issue.

## Running the tests

```bash
$ pip install "coniii[test]"
$ pytest --pyargs coniii
```

CI runs the tests on Linux, macOS and Windows for Python 3.9 through 3.13.

## Getting help and contributing

Please report bugs and request features on the
[issue tracker](https://github.com/eltrompetero/coniii/issues). Include your operating
system, your Python and ConIII versions (`python -c "import coniii; print(coniii.__version__)"`),
and the full error message or a short example that reproduces the problem. The output of
the tests above is often useful too.

Pull requests are welcome. Please open them against the `dev` branch. See
[DEVREADME](https://github.com/eltrompetero/coniii/blob/py3/DEVREADME) for how the
repository is organized.

## Release notes

Changes in each version are listed in
[RELEASE_NOTES](https://github.com/eltrompetero/coniii/blob/py3/RELEASE_NOTES). Version
4.0.0 changed the interface, so read its notes before upgrading from 3.x.

## Citing ConIII

If you use ConIII in your research, please cite:

> Lee, E.D. and Daniels, B.C., 2019. Convenient Interface to Inverse Ising (ConIII): A
> Python 3 Package for Solving Ising-Type Maximum Entropy Models. Journal of Open Research
> Software, 7(1), p.3. DOI: [10.5334/jors.217](https://doi.org/10.5334/jors.217).

```bibtex
@article{lee2019coniii,
  title   = {Convenient Interface to Inverse Ising (ConIII): A Python 3 Package for
             Solving Ising-Type Maximum Entropy Models},
  author  = {Lee, Edward D. and Daniels, Bryan C.},
  journal = {Journal of Open Research Software},
  volume  = {7},
  number  = {1},
  pages   = {3},
  year    = {2019},
  doi     = {10.5334/jors.217}
}
```

## License

ConIII is released under the MIT License. See
[LICENSE.txt](https://github.com/eltrompetero/coniii/blob/py3/LICENSE.txt).
