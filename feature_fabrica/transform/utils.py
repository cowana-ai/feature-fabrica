import re
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

DateTimeArray = Annotated[np.ndarray, Is[lambda array: array.dtype.type is np.datetime64]]
TimeDeltaArray = Annotated[np.ndarray, Is[lambda array: array.dtype.type is np.timedelta64]]

StrOrNumArray = Union[StrArray, NumericArray]

DATE_REGEX = re.compile(
    r'^(\d{4}-\d{2}-\d{2})'          # Matches YYYY-MM-DD
    r'(?:[ T](\d{2})(?::(\d{2}))?(?::(\d{2}))?)?$'  # Optionally matches HH, HH:MM, or HH:MM:SS
)

def is_numpy_datetime_format(date_str: str) -> bool:
    # Check if the string matches the common datetime formats (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    if not DATE_REGEX.match(date_str):
        return False
    return True

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
