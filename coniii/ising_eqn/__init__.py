"""Pre-generated symbolic equations for small Ising systems.

This subpackage holds machine-generated modules that encode the exact
probability distribution and observables for an Ising system of fixed
size N. Each module exposes a uniform interface (``p(hJ)``,
``calc_observables(hJ)``, etc.) tailored to a particular N.

File naming convention
----------------------
``ising_eqn_<N>.py``
    {0, 1} basis, double precision. N is the number of spins.
``ising_eqn_<N>_sym.py``
    {-1, +1} ("symmetric") basis, double precision.
``ising_eqn_<N>_sym_hp.py``
    {-1, +1} basis, arbitrary precision via :mod:`mpmath`.
``ising_eqn_<N>_triplet.py``, ``ising_eqn_<N>_sym_triplet.py``
    With third-order (triplet) interaction terms.

Regenerating
------------
Run :mod:`coniii.enumerate` directly to generate a new size, e.g.::

    python -m coniii.enumerate 7

The shell script ``write_ising_files.sh`` (in the package root)
regenerates the standard set (N=2..9 in both bases plus the N=5
triplet variants). High-precision (``_hp``) variants are produced by
passing the appropriate flag to :mod:`coniii.enumerate` — see that
module's docstring.
"""
