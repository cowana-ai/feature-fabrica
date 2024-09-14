from collections.abc import Iterable
from typing import Annotated, Union

import numpy as np
from beartype.vale import Is
from jaxtyping import Float, Integer

from feature_fabrica.models import FeatureValue

NumericArray = Union[Float[np.ndarray, "..."], Integer[np.ndarray, "..."]]
NumericValue = Union[np.floating, np.integer, float, int]

StrArray = Annotated[np.ndarray, Is[lambda array: array.dtype.kind == 'U']]
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
