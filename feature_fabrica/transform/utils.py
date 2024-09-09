from collections.abc import Iterable
from typing import Union

import numpy as np
from numpy.typing import NDArray

from feature_fabrica.models import FeatureValue

NumericArray = Union[Union[NDArray[np.float32], NDArray[np.float64]], Union[NDArray[np.int32], NDArray[np.int64]]]
NumericValue = Union[np.float32, np.float64, np.int32, np.int64, float, int]

StrArray = Union[NDArray[np.str_], np.ndarray]
StrValue = Union[np.str_, str]

StrOrNumArray = Union[StrArray, NumericArray]

def broadcast_and_normalize_numeric_array(iterable: Iterable) -> NumericArray:
    # Normalize all elements to np.array
    normalized_iterable = []
    for element in iterable:  # type: ignore[union-attr]
        if not isinstance(element, np.ndarray) and not isinstance(
            element, FeatureValue
        ):
            element = np.array([element], dtype=np.float32)
        normalized_iterable.append(element)
    # Find the maximum shape among the elements
    max_shape = np.broadcast_shapes(*[elem.shape for elem in normalized_iterable])

    # Broadcast elements to the maximum shape
    broadcasted_iterable = [
        np.broadcast_to(elem, max_shape) for elem in normalized_iterable
    ]
    return broadcasted_iterable
