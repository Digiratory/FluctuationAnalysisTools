import os

import pybind11
import pybind11.setup_helpers
from numpy import get_include
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Find pybind11 include directory
pybind11_include = pybind11.get_include()

# Original C API module - integrated into StatTools package
c_api_module = Extension(
    "StatTools.native.C_StatTools",
    include_dirs=[get_include()],
    sources=["src/cpp/StatTools_C_API.cpp"],
    language="c++",
)

# Modern pybind11 bindings - integrated into StatTools package
stattools_bindings = Extension(
    "StatTools.native.StatTools_bindings",
    include_dirs=[get_include(), pybind11_include],
    sources=["src/cpp/StatTools_bindings.cpp"],
    language="c++",
)

requirements = [line.strip() for line in open("requirements.txt").readlines()]

setup(
    ext_modules=[
        c_api_module,
        stattools_bindings,
    ],
    cmdclass={
        "build_ext": build_ext,
    },
    packages=[
        "StatTools",
        "StatTools.analysis",
        "StatTools.generators",
        "StatTools.filters",
    ],
    include_package_data=True,
    install_requires=requirements,
    description="A set of tools which allows to generate and process long-term dependent datasets",
)
