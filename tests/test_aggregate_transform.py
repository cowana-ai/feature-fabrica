import unittest

import numpy as np
from easydict import EasyDict as edict
from numpy.testing import assert_array_almost_equal, assert_array_equal

from feature_fabrica.models import FeatureValue
from feature_fabrica.transform import (ConcatenateReduce, GroupByReduce,
                                       MultiplyReduce, SubtractReduce,
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
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        expected = np.array([5, 5, 9, 9, 6])
        data = np.array([2, 3, 4, 5, 6])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', SumReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array(['1', '1', '2', '2', '3', '3']), data_type='str_')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array([2, 3, 4, 5, 6, 7])
        expected = np.array([5, 5, 9, 9, 13, 13])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

    def test_multiply_reduce(self):
        transform = GroupByReduce('feature', MultiplyReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        expected = np.array([6, 6, 20, 20, 42, 42])
        data = np.array([2, 3, 4, 5, 6, 7])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', MultiplyReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        expected = np.array([6, 6, 20, 20, 6])
        data = np.array([2, 3, 4, 5, 6])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

    def test_subtract_reduce(self):
        transform = GroupByReduce('feature', SubtractReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        expected = np.array([-1, -1, -1, -1, -1, -1])
        data = np.array([2, 3, 4, 5, 6, 7])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', SubtractReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        expected = np.array([-1, -1, -1, -1, 6])
        data = np.array([2, 3, 4, 5, 6])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

    def test_concat_reduce(self):
        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array(['1','1','2','2','3','3']), data_type='str_')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6','7'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '67', '67'])
        assert_array_equal(actual, expected)

        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array(['1','1','2','2','3']), data_type='str_')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '6'])
        assert_array_equal(actual, expected)

        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6','7'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '67', '67'])
        assert_array_equal(actual, expected)

    def test_common_strategy_reduce(self):
        for mode in ['mean', 'median']:
            transform = GroupByReduce('feature', mode, axis=-1)
            feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3, 3]), data_type='int32')
            example = edict({'feature': {'feature_value': feature_value}})
            transform.compile(example)
            data = np.array([5, 1, 9, 1, 13, 1])
            expected = np.array([3, 3, 5, 5, 7, 7])
            actual = transform.execute(data)
            assert_array_almost_equal(actual, expected)

            transform = GroupByReduce('feature', mode, axis=-1)
            feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 2, 2, 3]), data_type='int32')
            example = edict({'feature': {'feature_value': feature_value}})
            transform.compile(example)
            data = np.array([5, 1, 9, 1, 5, 3, 13])
            if mode == 'mean':
                expected = np.array([3, 3, 4.5, 4.5, 4.5, 4.5, 13])
            else:
                expected = np.array([3, 3, 4, 4, 4, 4, 13])
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

        transform = GroupByReduce('feature', 'max', axis=-1)
        feature_value = FeatureValue(value=np.array([1, 1, 2, 2, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array([5, 1, 9, 1, 13])
        expected = np.array([5, 5, 9, 9, 13])
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

        transform = GroupByReduce('feature', 'mode', axis=-1)
        feature_value = FeatureValue(value=np.array([1, 1, 1, 3, 3]), data_type='int32')
        example = edict({'feature': {'feature_value': feature_value}})
        transform.compile(example)
        data = np.array([5, 1, 5, 1, 13])
        expected = np.array([5, 5, 5, 1, 1])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
