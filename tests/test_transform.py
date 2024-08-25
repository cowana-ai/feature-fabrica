# test_transform.py
import unittest
from feature_fabrica.transform import scale_feature


class TestTransformations(unittest.TestCase):
    def test_scale_feature(self):
        self.assertEqual(scale_feature(10, 0.5), 5)


if __name__ == "__main__":
    unittest.main()
