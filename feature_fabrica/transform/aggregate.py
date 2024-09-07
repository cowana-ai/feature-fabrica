import numpy as np
from beartype import beartype

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import StrOrNumArray


class GroupByReduce(Transformation):
    @beartype
    def __init__(self, key_feature: str, reduce_func: Transformation, axis: int = 0):

        self.key_feature = key_feature
        self.reduce_func = reduce_func
        self.axis = axis
    @beartype
    def execute(self, data: StrOrNumArray) -> StrOrNumArray:

        #assert isinstance(self.key_feature, FeatureValue), "key_feature must be an existing feature!"
        assert self.key_feature.shape == data.shape # type: ignore[attr-defined]
        key_feature = self.key_feature

        sorted_idxs = key_feature.argsort() # type: ignore[attr-defined]
        data_sorted = data[sorted_idxs]
        key_feature_sorted = key_feature[sorted_idxs]

        data_aggregated =  np.split(data_sorted, np.unique(key_feature_sorted, return_index=True)[1][1:])

        data_aggregated = np.array(data_aggregated, dtype=data.dtype)
        data_reduced = self.reduce_func.execute(data_aggregated)

        # Use np.unique to find the unique key values and the counts for each
        _, inverse_indices = np.unique(key_feature, return_inverse=True)

        # Use np.repeat to assign the reduced values to the original positions
        result = data_reduced[inverse_indices]

        return result
