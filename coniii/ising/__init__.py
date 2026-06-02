"""Convenience helpers for Ising-model simulation.

This subpackage collects higher-level utilities specific to Ising
*simulation* (as opposed to the inverse-Ising inference machinery in
the top-level :mod:`coniii.solvers`). It is kept separate so that the
maximum-entropy solver code stays independent of any particular
lattice or graph structure.

Modules
-------
:mod:`coniii.ising.utils`
    General-purpose helpers including the convenience
    :class:`Ising` class. Re-exported here via wildcard import so
    that ``from coniii.ising import Ising`` works.
:mod:`coniii.ising.automaton`
    2D periodic-lattice ferromagnetic Ising simulator with quenched
    disorder. Import :class:`Ising2D` directly:
    ``from coniii.ising.automaton import Ising2D``.
"""
from .utils import *
