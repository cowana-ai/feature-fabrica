from collections.abc import Iterable
from functools import reduce

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import StrArray, StrValue


class ToLower(Transformation):
    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.char.lower(data)


class ToUpper(Transformation):
    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.char.upper(data)

class ConcatenateReduce(Transformation):
    def __init__(self, iterable: Iterable | None = None, expects_data: bool = False, axis: int=-1):
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
    def default(self) -> StrArray:
        return reduce(np.char.add, self.iterable) # type: ignore[arg-type]
    @beartype
    def with_data(self, data: StrArray) -> StrArray:
        return np.apply_along_axis(lambda x: reduce(np.char.add, x), axis=-1, arr=data)
    @beartype
    def with_data_and_iterable(self, data: StrArray) -> StrArray:
        # TODO: make the order configurable?
        iterable_with_data = [data] + self.iterable # type: ignore[operator]
        return reduce(np.char.add, iterable_with_data)

class Strip(Transformation):
    def __init__(self, chars: str | None = None):
        self.chars = chars

    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.char.strip(data, chars=self.chars)


class Split(Transformation):
    def __init__(self, delimiter: str):
        self.delimiter = delimiter

    @beartype
    def execute(self, data: StrArray | StrValue) -> np.ndarray:
        return np.char.split(data, self.delimiter)


class OneHotEncode(Transformation):
    def __init__(self, categories: list[str]):
        self.categories = categories
        self.encoder = OneHotEncoder(dtype=np.int32)
        self.encoder.fit([[category] for category in categories])

    @beartype
    def execute(self, data: StrArray | StrValue) -> NDArray[np.int32]:
        if isinstance(data, str):
            data = np.array([data])
        # Reshape the input data to a 2D array
        data_reshaped = data.reshape(-1, 1)  # type: ignore[union-attr]

        # Transform the data using the fitted encoder
        one_hot = self.encoder.transform(data_reshaped)
        return one_hot.toarray()


class LabelEncode(Transformation):
    def __init__(self, categories: list[str]):
        self.categories = categories
        self.encoder = LabelEncoder()
        self.encoder.fit(categories)

    @beartype
    def execute(self, data: StrArray | StrValue) -> NDArray[np.int32]:
        if isinstance(data, str):
            data = np.array([data])

        # Transform the data using the fitted encoder
        labels = self.encoder.transform(data)
        return labels.astype(np.int32)
