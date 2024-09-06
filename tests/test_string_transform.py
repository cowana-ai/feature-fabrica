import unittest

import numpy as np
from numpy.testing import assert_array_equal

from feature_fabrica.transform import (LabelEncode, OneHotEncode, Split, Strip,
                                       ToLower, ToUpper)


class TestStringTransformations(unittest.TestCase):
    def test_to_lower(self):
        transform = ToLower()
        data = np.array(["HELLO", "WORLD"])
        result = transform.execute(data)
        expected = np.array(["hello", "world"])
        assert_array_equal(result, expected)

        # Test with single string
        data_single = "HELLO"
        result_single = transform.execute(data_single)
        expected_single = "hello"
        self.assertEqual(result_single, expected_single)

    def test_to_upper(self):
        transform = ToUpper()
        data = np.array(["hello", "world"])
        result = transform.execute(data)
        expected = np.array(["HELLO", "WORLD"])
        assert_array_equal(result, expected)

        # Test with single string
        data_single = "hello"
        result_single = transform.execute(data_single)
        expected_single = "HELLO"
        self.assertEqual(result_single, expected_single)

    def test_strip(self):
        transform = Strip()
        data = np.array(["  hello  ", "  world  "])
        result = transform.execute(data)
        expected = np.array(["hello", "world"])
        assert_array_equal(result, expected)
        
        transform = Strip(chars=".")
        data = np.array([".hello.", "world."])
        result = transform.execute(data)
        expected = np.array(["hello", "world"])
        assert_array_equal(result, expected)

        # Test with single string
        transform = Strip()
        data_single = "  hello  "
        result_single = transform.execute(data_single)
        expected_single = "hello"
        self.assertEqual(result_single, expected_single)

    def test_split(self):
        transform = Split(delimiter="-")
        data = np.array(["hello-world", "foo-bar-sex"])
        result = transform.execute(data)
        expected = np.array([["hello", "world"], ["foo", "bar", "sex"]], dtype=object)
        assert_array_equal(result, expected)

    def test_one_hot_encode(self):
        categories = ["apple", "banana", "orange"]
        transform = OneHotEncode(categories)
        data = np.array(["apple", "orange"])
        result = transform.execute(data)
        expected = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.int32)
        assert_array_equal(result, expected)

        # Test with single string
        data_single = "banana"
        result_single = transform.execute(data_single)
        expected_single = np.array([[0, 1, 0]], dtype=np.int32)
        assert_array_equal(result_single, expected_single)

    def test_label_encode(self):
        categories = ["apple", "banana", "orange"]
        transform = LabelEncode(categories)
        data = np.array(["apple", "orange"])
        result = transform.execute(data)
        expected = np.array([0, 2], dtype=np.int32)
        assert_array_equal(result, expected)

        # Test with single string
        data_single = "banana"
        result_single = transform.execute(data_single)
        expected_single = np.array([1], dtype=np.int32)
        assert_array_equal(result_single, expected_single)


if __name__ == "__main__":
    unittest.main()
