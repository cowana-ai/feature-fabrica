from collections.abc import Iterable
from typing import Union

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from feature_fabrica.transform.base import Transformation

NumericArray = Union[NDArray[np.floating], NDArray[np.int_]]
NumericValue = Union[np.floating, np.int_, float, int]


class BaseReduce(Transformation):
    def __init__(self, iterable: Iterable | None = None, axis: int = 0):
        super().__init__()
        self.iterable = iterable
        self.axis = axis
        if self.iterable:
            self.execute = self.default  # type: ignore[method-assign]
        else:
            self.execute = self.with_data  # type: ignore[method-assign]

    def default(self):
        raise NotImplementedError()

    def with_data(self, data: NumericArray) -> NumericArray | NumericValue:
        raise NotImplementedError()


class SumReduce(BaseReduce):
    @beartype
    def default(self) -> NumericArray | NumericValue:
        # Normalize all elements to np.array
        normalized_iterable = []
        for element in self.iterable:  # type: ignore[union-attr]
            if not isinstance(element, np.ndarray):
                element = np.array([element], dtype=np.float32)
            normalized_iterable.append(element)
        # Find the maximum shape among the elements
        max_shape = np.broadcast_shapes(*[elem.shape for elem in normalized_iterable])

        # Broadcast elements to the maximum shape
        broadcasted_iterable = [
            np.broadcast_to(elem, max_shape) for elem in normalized_iterable
        ]
        return np.add.reduce(broadcasted_iterable, axis=self.axis)

    @beartype
    def with_data(self, data: NumericArray) -> NumericArray | NumericValue:
        return np.add.reduce(data, axis=self.axis)


class MultiplyReduce(BaseReduce):
    @beartype
    def default(self) -> NumericArray | NumericValue:
        # Normalize all elements to np.array
        normalized_iterable = []
        for element in self.iterable:  # type: ignore[union-attr]
            if not isinstance(element, np.ndarray):
                element = np.array([element], dtype=np.float32)
            normalized_iterable.append(element)
        # Find the maximum shape among the elements
        max_shape = np.broadcast_shapes(*[elem.shape for elem in normalized_iterable])

        # Broadcast elements to the maximum shape
        broadcasted_iterable = [
            np.broadcast_to(elem, max_shape) for elem in normalized_iterable
        ]
        return np.multiply.reduce(broadcasted_iterable, axis=self.axis)

    @beartype
    def with_data(self, data: NumericArray) -> NumericArray | NumericValue:
        return np.multiply.reduce(data, axis=self.axis)


class DivideTransform(Transformation):
    def __init__(
        self,
        numerator: str | float | None = None,
        denominator: str | float | None = None,
    ):
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
        return np.pow(data, self.power)


class ZScoreTransform(Transformation):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return (data - self.mean) / self.std_dev


class ClipTransform(Transformation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.clip(data, self.min, self.max)


class MinMaxTransform(Transformation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return (data - self.min) / (self.max - self.min)
