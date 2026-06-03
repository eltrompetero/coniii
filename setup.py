"""Build glue for the optional Boost.Python C++ extension.

All package metadata lives in pyproject.toml. This file exists only
because the package ships an optional C++ extension
(``coniii.samplers_ext``) and setuptools picks the Extension up from
setup.py during the PEP 517 build. If Boost is not available on the
build machine, the extension is skipped cleanly and the pure-Python
sampler implementations are used at runtime.

Boost's Python/NumPy libraries are named in several ways depending on
the distribution (``libboost_python313.so``, ``libboost_python.so``,
``libboost_python3.so``, versioned suffixes like
``libboost_python313.so.1.91.0``, ``.dylib`` on macOS, ...). Rather
than hard-coding one convention we discover the actual files on disk
and derive the correct link names from them.
"""
import glob
import os
import shutil
import sys

from setuptools import setup
from setuptools.extension import Extension


# Directories to search for Boost shared libraries, in priority order.
def _candidate_library_dirs(conda_prefix):
    dirs = []
    if conda_prefix:
        dirs.append(os.path.join(conda_prefix, "lib"))
    dirs += [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ]
    return [d for d in dirs if os.path.isdir(d)]


def _find_boost_lib(library_dirs, component):
    """Return the linker name for ``libboost_<component>*`` if found.

    e.g. a file ``libboost_python313.so.1.91.0`` yields the link name
    ``boost_python313``. Prefers a library whose name carries the
    running Python's version tag (``313``) so we don't accidentally
    pick up a Python 2 build. Returns ``None`` if nothing matches.
    """
    pyver = f"{sys.version_info.major}{sys.version_info.minor}"
    patterns = (f"libboost_{component}*.so*", f"libboost_{component}*.dylib")

    matches = []
    for d in library_dirs:
        for pat in patterns:
            matches.extend(glob.glob(os.path.join(d, pat)))
    if not matches:
        return None

    def link_name(path):
        base = os.path.basename(path)
        base = base[len("lib"):] if base.startswith("lib") else base
        for ext in (".so", ".dylib"):
            i = base.find(ext)
            if i != -1:
                base = base[:i]
                break
        return base

    names = [link_name(m) for m in matches]
    # Prefer an exact py-version match, then anything.
    for n in names:
        if pyver in n:
            return n
    return names[0]


def _detect_boost():
    """Probe for a usable Boost installation.

    Returns ``(include_dirs, library_dirs, extra_compile_args,
    boost_libs)`` when a usable Boost (headers + Python and NumPy
    libraries) is found, otherwise ``None``.
    """
    include_dirs = ["./cpp"]
    library_dirs = []
    extra_compile_args = ["-std=c++14"]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    have_headers = False
    if conda_prefix and os.path.isdir(os.path.join(conda_prefix, "include", "boost")):
        include_dirs.append(os.path.join(conda_prefix, "include"))
        extra_compile_args.append(f"-I{os.path.join(conda_prefix, 'include')}")
        # Respect a compiler already configured (e.g. by conda activation);
        # otherwise fall back to a compiler shipped in the conda prefix so we
        # link against the same ABI as conda's Boost.
        for var, exe in (("CC", "gcc"), ("CXX", "g++")):
            if os.environ.get(var):
                continue
            candidate = os.path.join(conda_prefix, "bin", exe)
            if os.path.exists(candidate):
                os.environ[var] = candidate
        have_headers = True
    elif any(os.path.isdir(d) for d in ("/usr/include/boost", "/usr/local/include/boost")):
        have_headers = True

    if not have_headers:
        return None

    library_dirs = _candidate_library_dirs(conda_prefix)
    python_lib = _find_boost_lib(library_dirs, "python")
    numpy_lib = _find_boost_lib(library_dirs, "numpy")
    if python_lib is None or numpy_lib is None:
        return None

    return include_dirs, library_dirs, extra_compile_args, [python_lib, numpy_lib]


# Ship the LICENSE inside the package so it lands in the wheel.
if os.path.exists("LICENSE.txt"):
    shutil.copyfile("LICENSE.txt", "coniii/LICENSE.txt")


ext_modules = []
boost = _detect_boost()
if boost is None:
    print("*" * 60)
    print("Boost (headers + Python/NumPy libraries) not detected.")
    print("Building coniii without the C++ samplers extension;")
    print("the pure-Python samplers will be used at runtime.")
    print("*" * 60)
else:
    include_dirs, library_dirs, extra_compile_args, boost_libs = boost
    print("*" * 60)
    print(f"Boost detected. Linking coniii.samplers_ext against: {boost_libs}")
    print("*" * 60)
    ext_modules.append(
        Extension(
            "coniii.samplers_ext",
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            sources=["./cpp/samplers.cpp", "./cpp/py.cpp"],
            libraries=boost_libs,
            extra_compile_args=extra_compile_args,
            language="c++",
        )
    )

setup(ext_modules=ext_modules)
