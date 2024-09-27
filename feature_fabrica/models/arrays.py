import numpy as np


class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
    """Mixin class to enable NumPy-like operations for custom models."""

    def _get_value(self):
        raise NotImplementedError()

    def _set_value(self, value: np.ndarray | None):
        raise NotImplementedError()

    def __array__(self, dtype=None, copy=None):
        """Automatic conversion to NumPy array when passed to NumPy functions."""
        if dtype:
            return self._get_value().astype(dtype)
        return self._get_value().copy() if copy else self._get_value()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle NumPy universal functions (e.g., np.add, np.multiply)."""
        inputs = tuple(x._get_value() if isinstance(x, ArrayLike) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return result

    def __array_function__(self, func, types, args, kwargs):
        """Handle array functions like np.mean, np.sum, etc."""
        args = tuple(x._get_value() if isinstance(x, ArrayLike) else x for x in args)
        result = func(*args, **kwargs)
        return result

    def __getitem__(self, idx):
        """Enable indexing like a NumPy array."""
        return self._get_value()[idx]

    def __getattr__(self, name):
        """Enable attribute access for underlying NumPy array properties."""
        return getattr(self._get_value(), name)
