import unittest

import numpy as np

from feature_fabrica.transform import AsType
from feature_fabrica.transform.utils import is_valid_numpy_dtype


class TestGenericTransform(unittest.TestCase):

    def test_astype(self):
        # Test for valid dtypes
        self.assertTrue(is_valid_numpy_dtype('int32'))
        self.assertTrue(is_valid_numpy_dtype('float64'))
        self.assertTrue(is_valid_numpy_dtype('datetime64[D]'))

        # Test for invalid dtype
        self.assertFalse(is_valid_numpy_dtype('invalid_dtype'))

        # Test valid data type conversion
        data = np.array([1.5, 2.6, 3.7], dtype='float64')
        astype_transformation = AsType(dtype='int32')

        # Execute transformation
        result = astype_transformation.execute(data)
        expected_result = data.astype('int32')

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected_result)

        with self.assertRaises(ValueError):
            AsType(dtype='invalid_dtype')

        # Test the execute method with valid dtype
        data = np.array([1.5, 2.6, 3.7], dtype='float64')
        astype_transformation = AsType(dtype='int32')

        result = astype_transformation.execute(data)
        expected_result = np.array([1, 2, 3], dtype='int32')

        np.testing.assert_array_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()
