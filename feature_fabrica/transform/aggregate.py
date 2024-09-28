from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from beartype import beartype
from scipy import stats

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import NumericArray, StrOrNumArray

if TYPE_CHECKING:
    from feature_fabrica.models import PromiseValue

@beartype
def mean_reduction(data: NumericArray | list[NumericArray], axis: int) -> NumericArray:
    if isinstance(data, np.ndarray):
        return np.mean(data, axis=axis)
    else:
        # Flatten the input arrays into a single contiguous array
        cells_flat = np.concatenate(data)

        # Compute the lengths and starting positions of each array
        cell_lengths = np.array([len(arr) for arr in data])
        cell_starts = np.insert(np.cumsum(cell_lengths[:-1]), 0, 0)
        return np.add.reduceat(cells_flat, cell_starts) / cell_lengths

@beartype
def min_reduction(data: NumericArray | list[NumericArray], axis: int) -> NumericArray:
    if isinstance(data, np.ndarray):
        return np.min(data, axis=axis)
    else:
        return np.array([np.min(arr) for arr in data], dtype=float)

@beartype
def max_reduction(data: NumericArray | list[NumericArray], axis: int) -> NumericArray:
    if isinstance(data, np.ndarray):
        return np.max(data, axis=axis)
    else:
        return np.array([np.max(arr) for arr in data], dtype=float)

@beartype
def median_reduction(data: NumericArray | list[NumericArray], axis: int) -> NumericArray:
    if isinstance(data, np.ndarray):
        return np.median(data, axis=axis)
    else:
        return np.array([np.median(arr) for arr in data], dtype=float)

@beartype
def mode_reduction(data: NumericArray | list[NumericArray], axis: int) -> NumericArray:
    if isinstance(data, np.ndarray):
        # use LabelEncode on your feature first: https://github.com/scipy/scipy/issues/15551
        mode_result = stats.mode(data, axis=axis)
        # Extract the mode from the result
        return mode_result.mode
    else:
        return np.array([stats.mode(arr, axis=-1).mode for arr in data], dtype=float)

DEAFULT_REDUCTIONS = dict(
        mean = mean_reduction,
        min = min_reduction,
        max = max_reduction,
        median = median_reduction,
        mode = mode_reduction
    )

class GroupByReduce(Transformation):
    _name_ = "group_reduce"
    @beartype
    def __init__(self, key_feature: str | PromiseValue, reduce_func: Transformation | str = "mean", axis: int = -1):
        super().__init__()
        if isinstance(reduce_func, str):
            assert reduce_func in ["mean", "mode", "min", "max", "median"]
            reduce_func = DEAFULT_REDUCTIONS[reduce_func]
        self.key_feature = key_feature
        self.reduce_func = reduce_func
        self.axis = axis

    @beartype
    def execute(self, data: StrOrNumArray) -> StrOrNumArray:
        """Groups data array by key_feature values and applies a reduction transformation.

        Assigns reduced values to the original positions to preserve the original shape.
        """
        if isinstance(self.reduce_func, Transformation):
            assert self.reduce_func.expects_data
        #assert isinstance(self.key_feature, FeatureValue), "key_feature must be an existing feature!"
        assert self.key_feature.shape == data.shape # type: ignore
        # FeatureValue has weird behavior with np.unique
        key_feature_value = self.key_feature.value # type: ignore

        sorted_idxs = key_feature_value.argsort() # type: ignore
        data_sorted = data[sorted_idxs]
        key_feature_sorted = key_feature_value[sorted_idxs] # type: ignore

        _, unqiue_indeces, counts = np.unique(key_feature_sorted, return_index=True, return_counts=True)
        data_aggregated =  np.split(data_sorted, unqiue_indeces[1:])

        if np.all(counts == counts[0]):
            data_aggregated = np.array(data_aggregated, dtype=data.dtype)

        if isinstance(self.reduce_func, Transformation):
            data_reduced = self.reduce_func.execute(data_aggregated)
        else:
            data_reduced = self.reduce_func(data_aggregated, self.axis) # type: ignore
        # Use np.unique to find the unique key values and the counts for each
        _, inverse_indices = np.unique(key_feature_value, return_inverse=True)
        # Use np.repeat to assign the reduced values to the original positions
        result = data_reduced[inverse_indices]

        return result
