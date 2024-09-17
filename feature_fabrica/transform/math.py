from collections.abc import Iterable

import numpy as np
from beartype import beartype
from sklearn.preprocessing import KBinsDiscretizer

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import (
    NumericArray, NumericValue, broadcast_and_normalize_numeric_array)


class BaseReduce(Transformation):
    ufunc = None
    def __init__(self, iterable: Iterable | None = None, expects_data: bool = False, axis: int = 0):
        super().__init__()

        assert iterable or expects_data, "Either expect_data or iterable should be set!"
        self.iterable = iterable
        self.axis = axis
        if not expects_data and self.iterable:
            self.execute = self.default  # type: ignore[method-assign]
        elif expects_data and not self.iterable:
            self.execute = self.with_data  # type: ignore[method-assign]
        elif expects_data and self.iterable:
            self.execute = self.with_data_and_iterable # type: ignore[method-assign]
    @beartype
    def default(self) -> NumericArray | NumericValue:
        if self.ufunc is None:
            raise NotImplementedError()
        iterable: NumericArray = broadcast_and_normalize_numeric_array(self.iterable)
        return self.ufunc.reduce(iterable, axis=self.axis)
    @beartype
    def with_data(self, data: NumericArray | list[NumericArray]) -> NumericArray | NumericValue:
        if self.ufunc is None:
            raise NotImplementedError()
        if isinstance(data, np.ndarray):
            return self.ufunc.reduce(data, axis=self.axis)
        else:
            # Flatten the input arrays into a single contiguous array
            cells_flat = np.concatenate(data, axis=self.axis)

            # Compute the lengths and starting positions of each array
            cell_lengths = np.array([len(arr) for arr in data])
            cell_starts = np.insert(np.cumsum(cell_lengths[:-1]), 0, 0, axis=self.axis)

            # Apply the reduceat function using the starting positions
            return self.ufunc.reduceat(cells_flat, cell_starts, axis=self.axis)

    @beartype
    def with_data_and_iterable(self, data: NumericArray) -> NumericArray | NumericValue:
        if self.ufunc is None:
            raise NotImplementedError()
        data_and_iterable: NumericArray = broadcast_and_normalize_numeric_array([data] + self.iterable)
        return self.ufunc.reduce(data_and_iterable, axis=self.axis)

class SumReduce(BaseReduce):
    ufunc = np.add

class MultiplyReduce(BaseReduce):
    ufunc = np.multiply

class SubtractReduce(BaseReduce):
    ufunc = np.subtract

class DivideTransform(Transformation):
    def __init__(
        self,
        numerator: str | float | None = None,
        denominator: str | float | None = None,
    ):
        super().__init__()
        assert (
            numerator or denominator
        ), "You have to pass either numerator or denominator for computation!"

        self.numerator = numerator
        self.denominator = denominator

        if numerator and denominator:
            self.execute = self.default  # type: ignore[method-assign]
        elif numerator:
            self.execute = self.with_numerator  # type: ignore[method-assign]
        else:
            self.execute = self.with_denominator  # type: ignore[method-assign]

    @beartype
    def with_numerator(
        self, data: NumericArray | NumericValue
    ) -> NumericArray | NumericValue:
        return self.numerator / data  # type: ignore[operator]

    @beartype
    def with_denominator(
        self, data: NumericArray | NumericValue
    ) -> NumericArray | NumericValue:
        return data / self.denominator  # type: ignore[operator]

    def default(self) -> NumericArray | NumericValue:
        return self.numerator / self.denominator  # type: ignore[operator]


class ScaleFeature(Transformation):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.multiply(data, self.factor)


class LogTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.log(data)


class ExpTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.exp(data)


class SqrtTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.sqrt(data)


class PowerTransform(Transformation):
    def __init__(self, power: float):
        self.power = power

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return data**self.power

class AbsoluteTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.abs(data)

class ZScoreTransform(Transformation):
    def __init__(self, mean: float | None = None, std_dev: float | None = None, axis: int = -1):
        super().__init__()
        self.mean = mean
        self.std_dev = std_dev
        self.axis = axis

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        if self.mean is not None and self.std_dev is not None:
            z_normalized = (data - self.mean) / self.std_dev
        else:
            assert isinstance(data, np.ndarray), "data must be array"
            # Calculate mean of the data
            mean = np.mean(data, axis=self.axis, keepdims=True)

            # Calculate the standard deviation of the data
            std_dev = np.std(data, axis=self.axis, keepdims=True)

            # Apply Z-score normalization
            z_normalized = (data - mean) / std_dev
        return z_normalized

class ClipTransform(Transformation):
    def __init__(self, min: float, max: float):
        super().__init__()
        self.min = min
        self.max = max

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.clip(data, self.min, self.max)


class MinMaxTransform(Transformation):
    def __init__(self, min: float | None = None, max: float | None = None, axis: int = -1):
        super().__init__()
        if min is not None and max is not None:
            assert min != max
        self.min = min
        self.max = max
        self.axis = axis

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        if self.min is not None and self.max is not None:
            min_max_normalized = (data - self.min) / (self.max - self.min)
        else:
            assert isinstance(data, np.ndarray), "data must be array"
            # Calculate min of the data
            min_ = np.min(data, axis=self.axis, keepdims=True)

            # Calculate max of the data
            max_ = np.max(data, axis=self.axis, keepdims=True)

            # Apply MinMax normalization
            min_max_normalized = (data - min_) / (max_ - min_)
        return min_max_normalized

class KBinsDiscretize(Transformation):
    @beartype
    def __init__(self,n_bins: int = 5, encode: str = 'onehot', strategy: str = 'quantile',  subsample: int | None = 200000, **kwargs):
        super().__init__()
        self.kbins = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample, **kwargs)

    @beartype
    def execute(self, data: NumericArray) -> NumericArray:
        shape = data.shape

        if len(shape) == 1:
            data = data.reshape(-1, 1)
        binned_data = self.kbins.fit_transform(data)
        if len(shape) == 1:
            binned_data = binned_data.reshape(shape)

        return binned_data
