#!/bin/bash

# Code for compiling package for uplaod to PyPI.
# Clean previous compilation results.
trash build dist

# Compile wheels into dist folder.
python setup.py bdist_wheel

# Rename Linux wheel for upload to PyPI.
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    rename 's/linux/manylinux1/' dist/*
fi

# Update usage guide to latest version for upload to PyPI.
cp ipynb/usage_guide.ipynb coniii/
