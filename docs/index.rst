.. ConIII documentation master file, created by
   sphinx-quickstart on Tue Apr 23 02:36:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ConIII's documentation!
==================================

ConIII (Convenient Interface to Inverse Ising) is a Python package for
solving maximum-entropy problems, with a focus on the pairwise (Ising)
inverse problem. See the project `README
<https://github.com/eltrompetero/coniii>`_ and ``usage_guide.ipynb``
for an introduction and worked examples.

Core API
--------

.. toctree::
   :maxdepth: 2
   :caption: Core API

   coniii_rst/coniii.solvers.rst
   coniii_rst/coniii.samplers.rst
   coniii_rst/coniii.models.rst
   coniii_rst/coniii.utils.rst

Enumeration and equations
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Enumeration

   coniii_rst/coniii.enumerate.rst
   coniii_rst/coniii.enumerate_potts.rst
   coniii_rst/coniii.ising.rst

Experimental and legacy
-----------------------

.. toctree::
   :maxdepth: 1
   :caption: Experimental & legacy

   coniii_rst/coniii.experimental.rst
   coniii_rst/coniii.legacy.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
