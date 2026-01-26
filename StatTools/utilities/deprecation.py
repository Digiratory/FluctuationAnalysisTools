"""
Deprecation utilities for StatTools package.
"""

import warnings


class DeprecatedWrapper:
    """
    A wrapper class to mark objects as deprecated.

    This wrapper will emit a deprecation warning when any attribute
    of the wrapped object is accessed.
    """

    def __init__(self, original, message, stacklevel=3):
        """
        Initialize the deprecated wrapper.

        Args:
            original: The original object to wrap
            message: The deprecation warning message
            stacklevel: The stack level for the warning (default: 3)
        """
        self._original = original
        self._message = message
        self._stacklevel = stacklevel

    def __getattr__(self, name):
        """Emit deprecation warning when any attribute is accessed."""
        warnings.warn(self._message, DeprecationWarning, stacklevel=self._stacklevel)
        return getattr(self._original, name)

    def __call__(self, *args, **kwargs):
        """Emit deprecation warning when the object is called."""
        warnings.warn(self._message, DeprecationWarning, stacklevel=self._stacklevel)
        return self._original(*args, **kwargs)
