# MIT License
# 
# Copyright (c) 2020 Edward D. Lee, Bryan C. Daniels
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#!/bin/bash

# Code for compiling package for upload to PyPI.
# Clean previous compilation results.
if [ ! `command -v trash` ]
then
    echo "trash-cli is not installed. Cannot empty dist directory properly."
else
    trash build dist
fi
find ./ -name *.pyc -exec rm {} \;

# Update cpp code
# rsync -au ../../cpp/cppsamplers/cppsamplers/*.*pp cpp/

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

if [ ! -d `has_dirs ./build/lib.*/coniii` ]; then
    echo "Failed to build Boost module."
fi
