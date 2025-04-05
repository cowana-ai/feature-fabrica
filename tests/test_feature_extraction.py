import unittest

import numpy as np

from feature_fabrica.transform import CountVectorize


class TestFeatureExtraction(unittest.TestCase):

    def test_count_vectorizer(self):
        # Test with a predefined vocabulary
        vocabulary = ['cat', 'dog', 'fish']
        transformer = CountVectorize(vocabulary=vocabulary)

        # Test single string input
        input_data = np.array(['dog and cat'])
        result = transformer.execute(input_data)

        expected = np.array([[1, 1, 0]])  # 'dog': 1, 'cat': 1, 'fish': 0
        np.testing.assert_array_equal(result, expected)

        # Test fitting without a predefined vocabulary
        transformer = CountVectorize(lowercase=True)

        input_data = np.array(['dog and cat', 'cat eats fish'])
        result = transformer.execute(input_data)

        self.assertEqual(result.shape, (2, 5))

        # Test when the input is a single string
        transformer = CountVectorize(vocabulary=['dog', 'cat'])

        input_data =  np.array(['dog'])
        result = transformer.execute(input_data)

        expected = np.array([[1, 0]])  # 'dog': 1, 'cat': 0
        np.testing.assert_array_equal(result, expected)

        # Test with empty input
        transformer = CountVectorize(vocabulary=['dog', 'cat'])

        input_data = np.array([''])
        result = transformer.execute(input_data)

        expected = np.array([[0, 0]])  # No tokens match
        np.testing.assert_array_equal(result, expected)

        # Test with no vocabulary (fitting from data)
        transformer = CountVectorize()

        input_data = np.array(['the dog barks'])
        result = transformer.execute(input_data)

        # Expect the vocabulary to be built from the input data
        self.assertEqual(result.shape[1], 3)  # 'the', 'dog', 'barks' are 3 unique tokens


if __name__ == '__main__':
    unittest.main()
