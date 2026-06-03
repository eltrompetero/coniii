"""Experimental / work-in-progress code.

These modules are intentionally **not** exported from the top-level
:mod:`coniii` package and their public API is unstable.  Every class
in :mod:`coniii.experimental.samplers` currently raises
:class:`NotImplementedError` on construction because the
implementations have not been validated end-to-end and are kept here
for future development.

Importing from this subpackage is intentionally explicit, e.g.::

    from coniii.experimental.samplers import SWIsing
"""
