import unittest

import numpy as np

from feature_fabrica.transform import DateTimeDifference


class TestDateTimeTransform(unittest.TestCase):

    def test_datetime_difference_valid(self):
        # Testing with valid initial datetime and data
        initial_datetime = '2023-01-01'
        data = np.array(['2023-01-05', '2023-01-10'], dtype=np.datetime64)
        transformation = DateTimeDifference(initial_datetime=initial_datetime, compute_unit='D')

        result = transformation.execute(data)

        expected = np.array([4, 9], dtype='timedelta64[D]')
        np.testing.assert_array_equal(result, expected)

        # Testing with valid end datetime and data
        end_datetime = '2023-01-10'
        data = np.array(['2023-01-05', '2023-01-06'], dtype=np.datetime64)
        transformation = DateTimeDifference(end_datetime=end_datetime, compute_unit='D')

        result = transformation.execute(data)

        expected = np.array([5, 4], dtype='timedelta64[D]')
        np.testing.assert_array_equal(result, expected)

        # Both initial_datetime and end_datetime should raise ValueError
        with self.assertRaises(ValueError) as context:
            DateTimeDifference(initial_datetime='2023-01-01', end_datetime='2023-01-05')
        self.assertIn("Only one of 'initial_datetime' or 'end_datetime' should be set!", str(context.exception))

        # Invalid compute_unit should raise ValueError
        with self.assertRaises(ValueError) as context:
            DateTimeDifference(initial_datetime='2023-01-01', compute_unit='invalid')
        self.assertIn("compute_unit= invalid is not a valid code!", str(context.exception))

        # Testing with conversion to a specific datetime format
        initial_datetime = '2023-01-01'
        data = np.array(['2023-01-05', '2023-01-10'], dtype=np.datetime64)
        transformation = DateTimeDifference(initial_datetime=initial_datetime, compute_unit='s')

        result = transformation.execute(data)

        expected = np.array([345600, 777600])
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
