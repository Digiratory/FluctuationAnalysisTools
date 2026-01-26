"""
StatTools - A comprehensive toolkit for long-term dependent datasets

This package provides tools for generating and processing long-term dependent datasets
with modern Python bindings for C++ implementations.
"""

# Import C++ bindings to make them available as part of the StatTools package
try:
    from .native import C_StatTools, StatTools_bindings
except ImportError:
    # C++ extensions not available, likely in development mode
    C_StatTools = None
    StatTools_bindings = None

# Mark C_StatTools as deprecated by wrapping it
from .utilities.deprecation import DeprecatedWrapper

if C_StatTools is not None:
    C_StatTools = DeprecatedWrapper(
        C_StatTools,
        "C_StatTools is deprecated and will be removed in version 2.0.0. "
        "Please use the Python-native implementations instead.",
    )

# Make key functions available directly from StatTools
if StatTools_bindings is not None:
    # Core functionality
    get_exponential_dist_value = StatTools_bindings.get_exponential_dist_value
    get_gauss_dist_value = StatTools_bindings.get_gauss_dist_value
    get_exp_dist_vector = StatTools_bindings.get_exp_dist_vector
    cumsum = StatTools_bindings.cumsum
    get_waiting_time = StatTools_bindings.get_waiting_time
    get_poisson_thread = StatTools_bindings.get_poisson_thread
    model = StatTools_bindings.model

# Package metadata
__author__ = "Aleksandr Sinitca, Alexandr Kuzmenko, Asya Lyanova"
__email__ = "amsinitca@etu.ru"
__description__ = (
    "A set of tools which allows to generate and process long-term dependent datasets"
)

# Public API - what users should import
__all__ = [
    # Analysis modules
    "analysis",
    "generators",
    "filters",
    "visualization",
    "Gamma",
    "auxiliary",
    # C++ bindings (if available)
    "C_StatTools",
    "StatTools_bindings",
    # Core functions (if bindings available)
    "get_exponential_dist_value",
    "get_gauss_dist_value",
    "get_exp_dist_vector",
    "cumsum",
    "get_waiting_time",
    "get_poisson_thread",
    "model",
]
