# MIT License
# 
# Copyright (c) 2017 Edward D. Lee, Bryan C. Daniels
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

# Code for compiling package for uplaod to PyPI.
# Clean previous compilation results.
trash build dist

# Update usage guide to latest version for upload to PyPI.
cp ipynb/usage_guide.ipynb coniii/

# Compile docs
sphinx-build ./docs/ ./docs/_build/html
rsync -avu docs/_build/html/* ~/Dropbox/Research/Documents/eltrompetero.github.io./coniii/

# Compile wheels into dist folder.
python setup.py bdist_wheel
# Make source available
python setup.py sdist

# Rename Linux wheel for upload to PyPI.
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    rename 's/linux/manylinux1/' dist/*
fi
