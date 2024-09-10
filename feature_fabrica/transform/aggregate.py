import numpy as np
from beartype import beartype
from scipy import stats

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import NumericArray, StrOrNumArray


@beartype
def mean_reduction(data: NumericArray, axis: int) -> NumericArray:
    return np.mean(data, axis=axis)

@beartype
def min_reduction(data: NumericArray, axis: int) -> NumericArray:
    return np.min(data, axis=axis)

@beartype
def max_reduction(data: NumericArray, axis: int) -> NumericArray:
    return np.max(data, axis=axis)

@beartype
def median_reduction(data: NumericArray, axis: int) -> NumericArray:
    return np.median(data, axis=axis)

@beartype
def mode_reduction(data: NumericArray, axis: int) -> NumericArray:
    # use LabelEncode on your feature first: https://github.com/scipy/scipy/issues/15551
    mode_result = stats.mode(data, axis=axis)
    # Extract the mode from the result
    return mode_result.mode

DEAFULT_REDUCTIONS = dict(
        mean = mean_reduction,
        min = min_reduction,
        max = max_reduction,
        median = median_reduction,
        mode = mode_reduction
    )

class GroupByReduce(Transformation):
    @beartype
    def __init__(self, key_feature: str, reduce_func: Transformation | str = "mean", axis: int = -1):

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
        assert self.key_feature.shape == data.shape # type: ignore[attr-defined]
        key_feature_value = self.key_feature.value # type: ignore[attr-defined]

        sorted_idxs = key_feature_value.argsort() # type: ignore[attr-defined]
        data_sorted = data[sorted_idxs]
        key_feature_sorted = key_feature_value[sorted_idxs]

        data_aggregated =  np.split(data_sorted, np.unique(key_feature_sorted, return_index=True)[1][1:])

        data_aggregated = np.array(data_aggregated, dtype=data.dtype)
        if isinstance(self.reduce_func, Transformation):
            data_reduced = self.reduce_func.execute(data_aggregated)
        else:
            data_reduced = self.reduce_func(data_aggregated, self.axis) # type: ignore[operator]
        # Use np.unique to find the unique key values and the counts for each
        _, inverse_indices = np.unique(key_feature_value, return_inverse=True)

        # Use np.repeat to assign the reduced values to the original positions
        result = data_reduced[inverse_indices]

        return result
