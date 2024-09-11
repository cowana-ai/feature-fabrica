from collections.abc import Iterable
from functools import reduce

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

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
    def with_data(self, data: StrArray | list[StrArray]) -> StrArray:
        if isinstance(data, np.ndarray):
            return np.apply_along_axis(lambda x: reduce(np.char.add, x), axis=-1, arr=data)
        else:
            return np.array([reduce(np.char.add, arr) for arr in data], dtype=str)

    @beartype
    def with_data_and_iterable(self, data: StrArray) -> StrArray:
        # TODO: make the order configurable?
        iterable_with_data = [data] + self.iterable # type: ignore[operator]
        return reduce(np.char.add, iterable_with_data)

class Strip(Transformation):
    def __init__(self, chars: str | None = None):
        super().__init__()
        self.chars = chars

    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.char.strip(data, chars=self.chars)


class Split(Transformation):
    def __init__(self, delimiter: str):
        super().__init__()
        self.delimiter = delimiter

    @beartype
    def execute(self, data: StrArray | StrValue) -> np.ndarray:
        return np.char.split(data, self.delimiter)


class OneHotEncode(Transformation):
    def __init__(self, categories: list[str] | None=None, **kwargs):
        super().__init__()
        self.encoder = OneHotEncoder(dtype=np.int32, **kwargs)
        self.categories = categories

        if self.categories is not None:
            self.categories = sorted(self.categories)
            self.encoder.fit([[category] for category in self.categories])

    @beartype
    def execute(self, data: StrArray | StrValue) -> NDArray[np.int32]:
        if isinstance(data, str):
            data = np.array([data])
        # Reshape the input data to a 2D array
        data_reshaped = data.reshape(-1, 1)  # type: ignore[union-attr]
        if self.categories is not None:
            # Transform the data using the fitted encoder
            one_hot = self.encoder.transform(data_reshaped)
        else:
            one_hot = self.encoder.fit_transform(data_reshaped)

        return one_hot.toarray()


class LabelEncode(Transformation):
    def __init__(self, categories: list[str] | None = None):
        super().__init__()
        self.encoder = LabelEncoder()
        self.categories = categories
        if self.categories is not None:
            self.categories = sorted(self.categories)
            self.encoder.fit(categories)

    @beartype
    def execute(self, data: StrArray | StrValue) -> NDArray[np.int32]:
        if isinstance(data, str):
            data = np.array([data])
        if self.categories is not None:
            # Transform the data using the fitted encoder
            labels = self.encoder.transform(data)
        else:
            labels = self.encoder.fit_transform(data)

        return labels.astype(np.int32)


class OrdinalEncode(Transformation):
    def __init__(self, categories: list[str] | None = None, **kwargs):
        super().__init__()
        self.encoder = OrdinalEncoder(dtype=np.int32, **kwargs)
        self.categories = categories

        if self.categories is not None:
            self.categories = sorted(self.categories)
            self.encoder.fit([[category] for category in self.categories])

    @beartype
    def execute(self, data: StrArray | StrValue) -> NDArray[np.int32]:
        if isinstance(data, str):
            data = np.array([data])
        data_reshaped = data.reshape(-1, 1)  # type: ignore[union-attr]
        if self.categories is not None:
            ordinals = self.encoder.transform(data_reshaped)
        else:
            ordinals = self.encoder.fit_transform(data_reshaped)

        return ordinals.ravel()


class BinaryEncode(Transformation):
    def __init__(self, categories: list[str] | None = None):
        super().__init__()
        self.categories = categories
        self._cached_dict = None

        if self.categories is not None:
            self.categories = np.array(sorted(self.categories))
            self._fit(self.categories)
            cached_dict = self._build_cache()
            self.transform = np.frompyfunc(lambda x:cached_dict[x], 1, 1)

    def _build_cache(self):
        if self._cached_dict is None:
            self._cached_dict = dict(zip(self.categories, self.binary_encoded_categories))

        return self._cached_dict

    def _fit(self, data: StrArray):
        # Get the unique values and their indices in the original data
        unique_vals, inverse_indices = np.unique(data, return_inverse=True)

        # Calculate the number of binary digits needed
        num_digits = len(bin(len(unique_vals) - 1)) - 2

        # Create the binary encoded array
        binary_encoded = ((inverse_indices[:, None] & (1 << np.arange(num_digits)[::-1])) > 0).astype(np.int32)

        self.binary_encoded_categories = binary_encoded

    @beartype
    def execute(self, data: StrArray) -> NDArray[np.int32]:
        """It is useful for reducing dimensionality when dealing with categorical features that would otherwise result
        in many one-hot encoded columns."""
        assert len(data.shape) == 1, "BinaryEncode expects 1D arrays"
        if self.categories is not None:
            binary_encoded = np.stack(self.transform(data), dtype=np.int32)
        else:
            # Get the unique values and their indices in the original data
            unique_vals, inverse_indices = np.unique(data, return_inverse=True)

            # Calculate the number of binary digits needed
            num_digits = len(bin(len(unique_vals) - 1)) - 2

            # Create the binary encoded array
            binary_encoded = ((inverse_indices[:, None] & (1 << np.arange(num_digits)[::-1])) > 0).astype(np.int32)

        return binary_encoded
