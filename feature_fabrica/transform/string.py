from typing import Union

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from feature_fabrica.transform.base import Transformation

StrArray = Union[NDArray[np.str_], np.ndarray]
StrValue = Union[np.str_, str]


class ToLower(Transformation):
    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.strings.lower(data)


class ToUpper(Transformation):
    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.strings.upper(data)


class Strip(Transformation):
    def __init__(self, chars: str | None = None):
        self.chars = chars

    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        return np.strings.strip(data)


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
