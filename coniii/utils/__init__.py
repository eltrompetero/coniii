"""Utility functions for the coniii package.

This subpackage was split out of a single monolithic ``utils.py`` to
keep concerns separate while preserving the public API: every name in
:data:`__all__` is still importable as ``coniii.utils.<name>`` and via
``from coniii.utils import *``.

Internal modules (leading underscore — do not import directly):

:mod:`coniii.utils._indexing`
    Coordinate <-> index helpers, state generation
    (``bin_states``, ``xbin_states``, ``xpotts_states``), base
    representation, matrix-vector shape helpers (``vec2mat``,
    ``mat2vec``, ``replace_diag``, ``zero_diag``).
:mod:`coniii.utils._correlations`
    ``pair_corr``, ``k_corr``, ``calc_de``, ``calc_overlap``,
    ``convert_corr``, ``state_probs``.
:mod:`coniii.utils._params`
    ``convert_params``, ``ising_convert_params``,
    ``split_concat_params``, and the ``define_*_helper_functions``
    factories.
:mod:`coniii.utils._graph`
    ``adj``, ``adj_sym``, ``coarse_grain_with_func``.
"""
from scipy.special import logsumexp

from ._indexing import (
    sub_to_ind, ind_to_sub,
    bin_states, xbin_states, xpotts_states,
    base_repr, unique_rows,
    vec2mat, mat2vec,
    replace_diag, zero_diag,
    # internal helpers used by other utils submodules and by tests
    unravel_index, multinomial,
)
from ._correlations import (
    pair_corr, k_corr, calc_de, calc_overlap,
    convert_corr, state_probs,
)
from ._params import (
    convert_params, ising_convert_params, split_concat_params,
    define_ising_helper_functions,
    define_ising_helper_functions_sym,
    define_potts_helper_functions,
    define_pseudo_ising_helper_functions,
    define_pseudo_potts_helper_functions,
    define_ternary_helper_functions,
    define_triplet_helper_functions,
    # internal helper, exposed because test_utils.py imports it
    _expand_binomial,
)
from ._graph import (
    adj, adj_sym,
    coarse_grain_with_func,
)


# Public API — same surface as the pre-split utils.py.
__all__ = [
    # indexing & state generation
    'sub_to_ind', 'ind_to_sub',
    'bin_states', 'xbin_states', 'xpotts_states',
    'base_repr', 'unique_rows',
    'vec2mat', 'mat2vec',
    # correlations
    'pair_corr', 'k_corr', 'calc_de', 'calc_overlap',
    'convert_corr', 'state_probs',
    # parameter conversion
    'convert_params', 'ising_convert_params', 'split_concat_params',
    # helper-function factories
    'define_ising_helper_functions',
    'define_ising_helper_functions_sym',
    'define_potts_helper_functions',
    'define_pseudo_ising_helper_functions',
    'define_pseudo_potts_helper_functions',
    'define_ternary_helper_functions',
    'define_triplet_helper_functions',
    # graph / matrix helpers
    'adj', 'adj_sym',
    'replace_diag', 'zero_diag',
    'coarse_grain_with_func',
    # numerical
    'logsumexp',
]
