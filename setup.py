# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
from os import path, environ
from distutils.extension import Extension
from coniii.version import version as __version__
from shutil import copyfile
import platform, sys

# set os environment default to be conda env path
os.environ['CC'] = os.environ['CONDA_PREFIX']+'/bin/'+os.environ['CC'].split('/')[-1]
os.environ['CXX'] = os.environ['CONDA_PREFIX']+'/bin/'+os.environ['CXX'].split('/')[-1]

# flags
NO_BOOST = False

# default args (that are modified per system specs below)
EXTRA_COMPILE_ARGS = ['-std=c++11',f'-I{os.environ["CONDA_PREFIX"]}/include']
if 'CONDA_PREFIX' in os.environ:
    # includes places to search for boost lib
    DEFAULT_LIBRARY_DR = [f'{os.environ["CONDA_PREFIX"]}/include']

# setup
here = path.abspath(path.dirname(__file__))
system = platform.system()
py_version = str(sys.version_info.major) + str(sys.version_info.minor)
dylibNames = [f'boost_python{py_version}', f'boost_numpy{py_version}']

# copy license into package
copyfile('LICENSE.txt','coniii/LICENSE.txt')

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# setup C++ extension
# make sure libraries exist if C++ extension is to be compiled
if not NO_BOOST:
    samplersModule = Extension('coniii.samplers_ext',
                               include_dirs = ['./cpp'],
                               library_dirs=DEFAULT_LIBRARY_DR,
                               sources=['./cpp/samplers.cpp', './cpp/py.cpp'],
                               extra_objects=['-l%s'%f for f in dylibNames],
                               extra_compile_args=EXTRA_COMPILE_ARGS,
                               language='c++')
    ext_modules = [samplersModule]
else:
    print("*******************************************")
    print("Boost not compiled because flag is not set.")
    print("*******************************************")


# compile
kwargs = {'name':'coniii',
          'version':__version__,
          'description':'Convenient Interface to Inverse Ising (ConIII)',
          'long_description':long_description,
          'long_description_content_type':'text/markdown',
          'url':'https://github.com/eltrompetero/coniii',
          'author':'Edward D. Lee, Bryan C Daniels',
          'author_email':'edlee@santafe.edu',
          'license':'MIT',
          'classifiers':['Development Status :: 5 - Production/Stable',
                         'Intended Audience :: Science/Research',
                         'Topic :: Scientific/Engineering :: Information Analysis',
                         'License :: OSI Approved :: MIT License',
                         'Programming Language :: Python :: 3 :: Only',
                        ],
          'python_requires':'>=3.8.3',
          'keywords':'inverse Ising maxent maximum entropy inference',
          'packages':find_packages(),
          'install_requires':['multiprocess>=0.70.7,<1',
                              'scipy',
                              'matplotlib',
                              'numpy>=1.16.2,<2',
                              'numba>=0.45.1,<1',
                              'mpmath>=1.1.0',
                              'dill'],
          'include_package_data':True,  # see MANIFEST.in
          'py_modules':['coniii.enumerate',
                        'coniii.enumerate_potts',
                        'coniii.mean_field_ising',
                        'coniii.pseudo_inverse_ising',
                        'coniii.samplers',
                        'coniii.solvers',
                        'coniii.utils'],
          'ext_modules':ext_modules}

try:
    setup(**kwargs)
except:
    print("*****************************************************")
    print("Boost not compiled. See above errors for g++ message.")
    print("*****************************************************")

    kwargs['ext_modules'] = []
    setup(**kwargs)
