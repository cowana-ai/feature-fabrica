import unittest

import numpy as np
from numpy.testing import assert_array_equal

from feature_fabrica.models import FeatureValue


class TestFeatureValue(unittest.TestCase):

    def test_math_operations(self):
        # Test addition
        feature_value = FeatureValue(value=np.array([2., 4.]), data_type='float32')
        result = np.add(feature_value, 2)
        expected = np.array([4., 6.], dtype='float32')
        assert_array_equal(result, expected)

        # Test multiplication
        result = np.multiply(feature_value, 2)
        expected = np.array([4., 8.], dtype='float32')
        assert_array_equal(result, expected)

        result = np.add.accumulate(feature_value)
        expected = np.array([2., 6.], dtype='float32')
        assert_array_equal(result, expected)

        # Test reduction
        result = np.multiply.reduce(feature_value)
        expected = np.array(8., dtype='float32')
        self.assertEqual(result, expected)

        # Test subtraction
        result = np.subtract(feature_value, 1)
        expected = np.array([1., 3.], dtype='float32')
        assert_array_equal(result, expected)

        # Test division
        result = np.divide(feature_value, 2)
        expected = np.array([1., 2.], dtype='float32')
        assert_array_equal(result, expected)

    def test_string_operations(self):
        # Test string concatenation
        array1 = np.array(['hello', 'good'], dtype='str')
        array2 = np.array([' there', 'bye'], dtype='str')
        feature_value = FeatureValue(value=array1, data_type='str_')
        result = np.char.add(feature_value, array2)
        expected = np.array(['hello there', 'goodbye'], dtype='str')
        assert_array_equal(result, expected)

        # Test with a transformation
        array3 = np.array(['!', ' now'], dtype='str')
        result = np.char.add(result, array3)
        expected = np.array(['hello there!', 'goodbye now'], dtype='str')
        assert_array_equal(result, expected)

        array1 = np.array(['Hello', 'Good'], dtype='str')
        feature_value = FeatureValue(value=array1, data_type='str_')
        result = np.char.lower(feature_value)
        expected = np.array(['hello', 'good'], dtype='str')
        assert_array_equal(result, expected)

        array1 = np.array(['Hello', 'Good'], dtype='str')
        feature_value = FeatureValue(value=array1, data_type='str_')
        result = np.char.upper(feature_value)
        expected = np.array(['HELLO', 'GOOD'], dtype='str')
        assert_array_equal(result, expected)

    def test_dtype_conversion(self):
        # Test dtype conversion
        feature_value = FeatureValue(value=np.array([2., 4.]), data_type='float32')
        result = np.array(feature_value, dtype='float64')
        expected = np.array([2., 4.], dtype='float64')
        assert_array_equal(result, expected)

        # Ensure no unintended mutation
        self.assertEqual(feature_value.value.dtype, np.float32)

    def test_array_interface(self):
        # Test the __array_interface__ attribute
        feature_value = FeatureValue(value=np.array([2., 4.]), data_type='float32')
        array_interface = feature_value.__array_interface__

        self.assertIsInstance(array_interface, dict)
        self.assertIn('shape', array_interface)
        self.assertIn('typestr', array_interface)
        self.assertIn('data', array_interface)
        self.assertEqual(array_interface['shape'], (2,))
        self.assertEqual(array_interface['shape'], feature_value.shape)
        self.assertEqual(array_interface['typestr'], '<f4')  # Little-endian float32

if __name__ == '__main__':
    unittest.main()
