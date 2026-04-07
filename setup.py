from numpy import get_include
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

# Original C API module - integrated into StatTools package
c_api_module = Extension(
    "StatTools.native.C_StatTools",
    include_dirs=[get_include()],
    sources=["src/cpp/StatTools_C_API.cpp", "src/cpp/StatTools_core.cpp"],
    language="c++",
    extra_compile_args=["-std=c++14"],
)

# Modern pybind11 bindings - integrated into StatTools package
stattools_bindings = Pybind11Extension(
    "StatTools.native.StatTools_bindings",
    include_dirs=[get_include()],
    sources=["src/cpp/StatTools_bindings.cpp", "src/cpp/StatTools_core.cpp"],
)

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
    description="A set of tools which allows to generate and process long-term dependent datasets",
)
