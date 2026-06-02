"""Build glue for the optional Boost.Python C++ extension.

All package metadata lives in pyproject.toml. This file exists only
because the package ships an optional C++ extension
(``coniii.samplers_ext``) and setuptools picks the Extension up from
setup.py during the PEP 517 build. If Boost is not available on the
build machine, the extension is skipped cleanly and the pure-Python
sampler implementations are used at runtime.
"""
import os
import shutil
import sys

from setuptools import setup
from setuptools.extension import Extension


def _detect_boost():
    """Probe for a usable Boost installation.

    Returns ``(include_dirs, library_dirs, extra_compile_args)`` if
    found, otherwise ``None``.
    """
    include_dirs = ["./cpp"]
    library_dirs = []
    extra_compile_args = ["-std=c++11"]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and os.path.isdir(os.path.join(conda_prefix, "include", "boost")):
        include_dirs.append(os.path.join(conda_prefix, "include"))
        library_dirs.append(os.path.join(conda_prefix, "lib"))
        extra_compile_args.append(f"-I{os.path.join(conda_prefix, 'include')}")
        # Prefer the conda toolchain so we link against conda's Boost ABI.
        for var, exe in (("CC", "gcc"), ("CXX", "g++")):
            candidate = os.path.join(conda_prefix, "bin", exe)
            if os.path.exists(candidate):
                os.environ[var] = candidate
        return include_dirs, library_dirs, extra_compile_args

    for sysdir in ("/usr/include/boost", "/usr/local/include/boost"):
        if os.path.isdir(sysdir):
            return include_dirs, library_dirs, extra_compile_args

    return None


# Ship the LICENSE inside the package so it lands in the wheel.
if os.path.exists("LICENSE.txt"):
    shutil.copyfile("LICENSE.txt", "coniii/LICENSE.txt")


ext_modules = []
boost = _detect_boost()
if boost is None:
    print("*" * 60)
    print("Boost not detected on this system.")
    print("Building coniii without the C++ samplers extension;")
    print("the pure-Python samplers will be used at runtime.")
    print("*" * 60)
else:
    include_dirs, library_dirs, extra_compile_args = boost
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    dylib_names = [f"boost_python{py_version}", f"boost_numpy{py_version}"]
    ext_modules.append(
        Extension(
            "coniii.samplers_ext",
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            sources=["./cpp/samplers.cpp", "./cpp/py.cpp"],
            extra_objects=[f"-l{lib}" for lib in dylib_names],
            extra_compile_args=extra_compile_args,
            language="c++",
        )
    )

setup(ext_modules=ext_modules)
