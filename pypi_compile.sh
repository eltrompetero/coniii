#!/bin/bash

# Code for compiling package for upload to PyPI.
# Clean previous compilation results.
if [ ! `command -v trash` ]
then
    echo "trash-cli is not installed. Cannot empty dist directory safely."
else
    if [ -d build ]; then
        trash build
    fi
    if [ -d dist ]; then
        trash dist
    fi
fi
find ./ -name *.pyc -exec rm {} \;

# Compile wheels into dist folder and make source available.
# Metadata is read from pyproject.toml; setup.py only declares the
# optional Boost C++ extension.
python -m build

# Rename Linux wheel for upload to PyPI.
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    if [ ! `command -v rename` ]
    then
        echo "rename is not installed. wheel not renamed."
        exit 1
    fi
    rename 's/linux/manylinux1/' dist/*
fi

# For pypi upload
if [ "$1" == "--all" ]
then
    # Update usage guide to latest version for upload to PyPI.
    cp ipynb/usage_guide.ipynb coniii/

    # Compile docs
    sphinx-build ./docs/ ./docs/_build/html

    # NOTE: the docs publish step was previously an rsync into a
    # personal Dropbox-backed clone of eltrompetero.github.io. That
    # path was specific to one machine and has been removed. See
    # DEVREADME for the current publish flow.
fi

# check if boost module compiled
has_dirs() {
  for f do
    [ -d "$f" ] && return
  done
  false
}

if compgen -G "./build/lib.*/coniii/samplers_ext*.so" > /dev/null; then
    echo "********************************"
    echo "Boost module built successfully."
    echo "********************************"
else
    echo "*****************************"
    echo "Failed to build Boost module."
    echo "*****************************"
fi
