"""Legacy code retained for backward compatibility.

The modules in this subpackage are kept because parts of the active
codebase still depend on them, but they have not been actively
maintained. Each module emits a :class:`DeprecationWarning` on
import.

Modules
-------
:mod:`coniii.legacy.mean_field_ising`
    Mean-field Ising solver helpers. Used internally by
    :class:`coniii.solvers.ClusterExpansion` and
    :class:`coniii.solvers.RegularizedMeanField`.
:mod:`coniii.legacy.pseudo_inverse_ising`
    Stand-alone pseudolikelihood inverse-Ising prototype. Not wrapped
    by any solver class and not exported from the top-level package.

These modules are scheduled for removal or rewrite in coniii v5.
"""
