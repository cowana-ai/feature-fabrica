import unittest

import numpy as np
from numpy.testing import assert_array_equal

from feature_fabrica.models import PromiseValue
from feature_fabrica.transform import (BinaryEncode, ConcatenateReduce,
                                       LabelEncode, OneHotEncode,
                                       OrdinalEncode, Split, Strip, ToLower,
                                       ToUpper)


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

    def test_concatenate_reduce(self):
        array1 = np.array(['hello', 'good'])
        array2 = np.array([' there', 'bye'])
        array3 = np.array(['!', ' now'])

        transform = ConcatenateReduce(iterable=[PromiseValue(value=array1, data_type='str_'), array2, array3])
        result = transform.execute()
        expected = np.array(['hello there!', 'goodbye now'])
        assert_array_equal(result, expected)

        stacked = np.stack([array1, array2, array3], axis=1)
        transform = ConcatenateReduce()
        result = transform.execute(stacked)
        expected = np.array(['hello there!', 'goodbye now'])
        assert_array_equal(result, expected)

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
        transform = OneHotEncode(categories=categories)
        data = np.array(["apple", "orange"])
        result = transform.execute(data)
        expected = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.int32)
        assert_array_equal(result, expected)
        # Test with single string
        data_single = "banana"
        result_single = transform.execute(data_single)
        expected_single = np.array([[0, 1, 0]], dtype=np.int32)
        assert_array_equal(result_single, expected_single)

        transform = OneHotEncode()
        data = np.array(["orange", "apple"])
        result = transform.execute(data)
        expected = np.array([[0, 1], [1, 0]], dtype=np.int32)
        assert_array_equal(result, expected)

        categories = ["apple", "banana", "orange"]
        transform = OneHotEncode(categories=categories, handle_unknown='ignore')
        data = np.array(["kiwi"])
        result = transform.execute(data)
        expected = np.array([[0, 0, 0]], dtype=np.int32)
        assert_array_equal(result, expected)


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

        transform = LabelEncode()
        data = np.array(["orange", "apple"])
        result = transform.execute(data)
        expected = np.array([1, 0], dtype=np.int32)
        assert_array_equal(result, expected)

    def test_ordinal_encode(self):
        categories = ["apple", "banana", "orange"]
        transform = OrdinalEncode(categories)
        data = np.array(["apple", "orange"])
        result = transform.execute(data)
        expected = np.array([0, 2], dtype=np.int32)
        assert_array_equal(result, expected)

        # Test with single string
        data_single = "banana"
        result_single = transform.execute(data_single)
        expected_single = np.array([1], dtype=np.int32)
        assert_array_equal(result_single, expected_single)

        transform = OrdinalEncode()
        data = np.array(["orange", "apple"])
        result = transform.execute(data)
        expected = np.array([1, 0], dtype=np.int32)
        assert_array_equal(result, expected)

    def test_binary_encode(self):
        transform = BinaryEncode()
        data = np.array(['red', 'blue', 'green', 'yellow'])
        result = transform.execute(data)
        expected = np.array([[1, 0],
                            [0, 0],
                            [0, 1],
                            [1, 1]], dtype=np.int32)
        assert_array_equal(result, expected)

        transform = BinaryEncode(categories=['red', 'blue', 'green', 'yellow'])
        data = np.array(['red', 'blue', 'green', 'yellow'])
        result = transform.execute(data)
        expected = np.array([[1, 0],
                            [0, 0],
                            [0, 1],
                            [1, 1]], dtype=np.int32)
        assert_array_equal(result, expected)

        data = np.array(['blue', 'green'])
        result = transform.execute(data)
        expected = np.array([[0, 0], [0, 1]], dtype=np.int32)
        assert_array_equal(result, expected)




if __name__ == "__main__":
    unittest.main()
