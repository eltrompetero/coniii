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

# Update cpp code (DEPRECATED)
# rsync -au ../../cpp/cppsamplers/cppsamplers/*.*pp cpp/

# Apply current conda environment
conda init $CONDA_DEFAULT_ENV

# Compile wheels into dist folder.
python setup.py bdist_wheel
# Make source available
python setup.py sdist

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

    echo "rsync -au docs/_build/html/* ~/Dropbox/Documents/eltrompetero.github.io/coniii/"
    rsync -au docs/_build/html/* ~/Dropbox/Documents/eltrompetero.github.io/coniii/
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
