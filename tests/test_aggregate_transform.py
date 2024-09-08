import unittest

import numpy as np
from easydict import EasyDict as edict
from numpy.testing import assert_array_almost_equal, assert_array_equal

from feature_fabrica.transform import (ConcatenateReduce, GroupByReduce,
                                       SumReduce)


class TestTransformations(unittest.TestCase):
    def test_sum_reduce(self):
        transform = GroupByReduce('feature', SumReduce(expects_data=True, axis=-1))
        example = edict({'feature': {'feature_value': np.array([1,1,2,2,3,3])}})
        transform.compile(example)
        expected = np.array([ 5,  5,  9,  9, 13, 13])
        data = np.array([2,3,4,5,6,7])
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)


        transform = GroupByReduce('feature', SumReduce(expects_data=True, axis=-1))
        example = edict({'feature': {'feature_value': np.array(['1','1','2','2','3','3'])}})
        transform.compile(example)
        actual = transform.execute(data)
        assert_array_almost_equal(actual, expected)

        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        example = edict({'feature': {'feature_value': np.array(['1','1','2','2','3','3'])}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6','7'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '67', '67'])
        assert_array_equal(actual, expected)

        transform = GroupByReduce('feature', ConcatenateReduce(expects_data=True, axis=-1))
        example = edict({'feature': {'feature_value': np.array([1,1,2,2,3,3])}})
        transform.compile(example)
        data = np.array(['2','3','4','5','6','7'])
        actual = transform.execute(data)
        expected = np.array(['23', '23', '45', '45', '67', '67'])
        assert_array_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
