import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from feature_fabrica.transform import (ClipTransform, DivideTransform,
                                       ExpTransform, LogTransform,
                                       MinMaxTransform, MultiplyReduce,
                                       PowerTransform, ScaleFeature,
                                       SqrtTransform, SubtractReduce,
                                       SumReduce, ZScoreTransform)


class TestTransformations(unittest.TestCase):
    def test_sum_reduce(self):
        data = [np.array([1, 2, 3]), 4]
        transform = SumReduce(data)
        transform.compile()
        result = transform.execute()
        expected = np.array([5, 6, 7])
        assert_array_almost_equal(result, expected)

        transform = SumReduce(expects_data=True)
        result = transform.execute(np.array([1, 2, 3]))
        expected = np.array([6])
        assert_array_almost_equal(result, expected)

        transform = SumReduce(iterable=[np.array([1, 2, 3]), 4], expects_data=True)
        result = transform.execute(np.array([1]))
        expected = np.array([6, 7, 8])
        assert_array_almost_equal(result, expected)


    def test_multiply_reduce(self):
        data = [np.array([1, 2, 3]), 2]
        transform = MultiplyReduce(data)
        transform.compile()
        result = transform.execute()
        expected = np.array([2, 4, 6])
        assert_array_almost_equal(result, expected)

        transform = MultiplyReduce(expects_data=True)
        transform.compile()
        result = transform.execute(np.array([1, 2, 3]))
        expected = np.array([6])
        assert_array_almost_equal(result, expected)

        transform = MultiplyReduce(iterable=[np.array([1, 2, 3]), 4], expects_data=True)
        result = transform.execute(np.array([1]))
        expected = np.array([4, 8, 12])
        assert_array_almost_equal(result, expected)


    def test_subtract_reduce(self):
        data = [np.array([1, 2, 3]), 2]
        transform = SubtractReduce(data)
        transform.compile()
        result = transform.execute()
        expected = np.array([-1, 0, 1])
        assert_array_almost_equal(result, expected)

        transform = SubtractReduce(expects_data=True)
        transform.compile()
        result = transform.execute(np.array([1, 2, 3]))
        expected = np.array([-4])
        assert_array_almost_equal(result, expected)

        transform = SubtractReduce(iterable=[np.array([1, 2, 3]), 4], expects_data=True)
        result = transform.execute(np.array([1]))
        expected = np.array([-4, -5, -6])
        assert_array_almost_equal(result, expected)

    def test_divide_transform_with_numerator(self):
        transform = DivideTransform(numerator=10)
        transform.compile()
        data = np.array([2, 5, 10])
        result = transform.execute(data)
        expected = np.array([5, 2, 1])
        assert_array_almost_equal(result, expected)

    def test_divide_transform_with_denominator(self):
        transform = DivideTransform(denominator=10)
        transform.compile()
        data = np.array([20, 50, 100])
        result = transform.execute(data)
        expected = np.array([2, 5, 10])
        assert_array_almost_equal(result, expected)

    def test_divide_transform_default(self):
        transform = DivideTransform(numerator=100, denominator=20)
        transform.compile()
        result = transform.execute()
        expected = 5
        self.assertEqual(result, expected)

    def test_scale_feature(self):
        transform = ScaleFeature(factor=2.5)
        transform.compile()
        data = np.array([1, 2, 3])
        result = transform.execute(data)
        expected = np.array([2.5, 5, 7.5])
        assert_array_almost_equal(result, expected)

    def test_log_transform(self):
        transform = LogTransform()
        transform.compile()
        data = np.array([1, np.e, np.e**2])
        result = transform.execute(data)
        expected = np.array([0, 1, 2])
        assert_array_almost_equal(result, expected)

    def test_exp_transform(self):
        transform = ExpTransform()
        transform.compile()
        data = np.array([0, 1, 2])
        result = transform.execute(data)
        expected = np.array([1, np.e, np.e**2])
        assert_array_almost_equal(result, expected)

    def test_sqrt_transform(self):
        transform = SqrtTransform()
        transform.compile()
        data = np.array([1, 4, 9])
        result = transform.execute(data)
        expected = np.array([1, 2, 3])
        assert_array_almost_equal(result, expected)

    def test_power_transform(self):
        transform = PowerTransform(power=3)
        transform.compile()
        data = np.array([1, 2, 3])
        result = transform.execute(data)
        expected = np.array([1, 8, 27])
        assert_array_almost_equal(result, expected)

    def test_zscore_transform(self):
        transform = ZScoreTransform(mean=5, std_dev=2)
        transform.compile()
        data = np.array([3, 5, 7])
        result = transform.execute(data)
        expected = np.array([-1, 0, 1])
        assert_array_almost_equal(result, expected)

        transform = ZScoreTransform()
        transform.compile()
        data = np.array([3, 5, 7])
        data_mean = np.mean(data)
        data_std_dev = np.std(data)
        result = transform.execute(data)
        expected = (data - data_mean) / data_std_dev
        assert_array_almost_equal(result, expected)

        data = np.array([[3, 5, 7], [1, 2, 3]])
        data_mean = np.mean(data, axis=-1, keepdims=True)
        data_std_dev = np.std(data, axis=-1, keepdims=True)
        result = transform.execute(data)
        expected = (data - data_mean) / data_std_dev
        assert_array_almost_equal(result, expected)

    def test_clip_transform(self):
        transform = ClipTransform(min=0, max=10)
        transform.compile()
        data = np.array([-5, 5, 15])
        result = transform.execute(data)
        expected = np.array([0, 5, 10])
        assert_array_almost_equal(result, expected)

    def test_minmax_transform(self):
        transform = MinMaxTransform(min=0, max=10)
        transform.compile()
        data = np.array([0, 5, 10])
        result = transform.execute(data)
        expected = np.array([0, 0.5, 1])
        assert_array_almost_equal(result, expected)

        transform = MinMaxTransform()
        transform.compile()
        data = np.array([0, 5, 10])
        result = transform.execute(data)
        expected = np.array([0, 0.5, 1])
        assert_array_almost_equal(result, expected)

        data = np.array([[0, 5, 10], [1, 2, 3]])
        result = transform.execute(data)
        expected = np.array([[0, 0.5, 1], [0, 0.5, 1]])
        assert_array_almost_equal(result, expected)



if __name__ == "__main__":
    unittest.main()
