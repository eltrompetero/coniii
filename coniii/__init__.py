"""Convenient Interface to Inverse Ising (ConIII).

A Python package for solving maximum entropy problems with a focus on
the pairwise (Ising) inverse problem. See the README and
``usage_guide.ipynb`` for documentation and examples.

The top-level namespace re-exports the solver classes (used through
``from coniii import MCH`` etc.) and the most common utility
functions. Sampler classes live in :mod:`coniii.samplers`, model
classes in :mod:`coniii.models`, and the exact symbolic equations for
small Ising systems in :mod:`coniii.ising_eqn`.
"""
from .solvers import (
    Enumerate,
    SparseEnumerate,
    MPF,
    MCH,
    Pseudo,
    ClusterExpansion,
    RegularizedMeanField,
)
from .utils import (
    # indexing & state generation
    sub_to_ind, ind_to_sub,
    bin_states, xbin_states, xpotts_states,
    base_repr, unique_rows,
    vec2mat, mat2vec,
    # correlations
    pair_corr, k_corr, calc_de, calc_overlap,
    convert_corr, state_probs,
    # parameter conversion
    convert_params, ising_convert_params, split_concat_params,
    # helper-function factories
    define_ising_helper_functions,
    define_ising_helper_functions_sym,
    define_potts_helper_functions,
    define_pseudo_ising_helper_functions,
    define_pseudo_potts_helper_functions,
    define_ternary_helper_functions,
    define_triplet_helper_functions,
    # graph / matrix helpers
    adj, adj_sym,
    replace_diag, zero_diag,
    coarse_grain_with_func,
    # numerical
    logsumexp,
)
from .version import version as __version__


__all__ = [
    '__version__',
    # Solvers (also accessible via `coniii.solvers`)
    'Enumerate', 'SparseEnumerate',
    'MPF',
    'MCH',
    'Pseudo',
    'ClusterExpansion',
    'RegularizedMeanField',
    # Utilities (also accessible via `coniii.utils`)
    'sub_to_ind', 'ind_to_sub',
    'bin_states', 'xbin_states', 'xpotts_states',
    'base_repr', 'unique_rows',
    'vec2mat', 'mat2vec',
    'pair_corr', 'k_corr', 'calc_de', 'calc_overlap',
    'convert_corr', 'state_probs',
    'convert_params', 'ising_convert_params', 'split_concat_params',
    'define_ising_helper_functions',
    'define_ising_helper_functions_sym',
    'define_potts_helper_functions',
    'define_pseudo_ising_helper_functions',
    'define_pseudo_potts_helper_functions',
    'define_ternary_helper_functions',
    'define_triplet_helper_functions',
    'adj', 'adj_sym',
    'replace_diag', 'zero_diag',
    'coarse_grain_with_func',
    'logsumexp',
]
