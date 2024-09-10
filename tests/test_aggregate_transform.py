import unittest

import numpy as np
from easydict import EasyDict as edict
from numpy.testing import assert_array_almost_equal, assert_array_equal

from feature_fabrica.models import FeatureValue
from feature_fabrica.transform import (ConcatenateReduce, GroupByReduce,
                                       SumReduce)


class TestTransformations(unittest.TestCase):
    def test_sum_reduce(self):
        transform = GroupByReduce('feature', SumReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        expected = np.array([5, 5, 9, 9, 13, 13])
        data = np.array([2, 3, 4, 5, 6, 7])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)


        transform = GroupByReduce('feature', SumReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array(['1', '1', '2', '2', '3', '3']), data_type='str_')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array(['1','1','2','2','3','3']), data_type='str_')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6','7'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '67', '67'])
        assert_array_equal(actual, expected)

        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6','7'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '67', '67'])
        assert_array_equal(actual, expected)

        for mode in ['mean', 'median']:
            transform = GroupByReduce('feature', mode, axis=-1)
            feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
            example = edict({'feature': {'feature_value': feature_value}})
            transform.compile(example)
            data = np.array([5, 1, 9, 1, 13, 1])
            expected = np.array([3, 3, 5, 5, 7, 7])
            actual = transform.execute(data)
            assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', 'max', axis=-1)
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array([5, 1, 9, 1, 13, 1])
        expected = np.array([5, 5, 9, 9, 13, 13])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', 'mode', axis=-1)
        feature_value = FeatureValue(value=np.array([1, 1, 1, 3, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array([5, 1, 5, 1, 13, 1])
        expected = np.array([5, 5, 5, 1, 1, 1])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
