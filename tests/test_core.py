# test_core.py
import unittest
from feature_fabrica.core import FeatureSet


class TestFeatureSet(unittest.TestCase):
    def test_load_features(self):
        feature_set = FeatureSet("examples/basic_features.yaml")
        self.assertIn("feature_a", feature_set.features)
        self.assertIn("feature_c", feature_set.features)

    def test_compute_all(self):
        data = {"feature_a": 10, "feature_b": 20}
        feature_set = FeatureSet("examples/basic_features.yaml")
        results = feature_set.compute_all(data)
        self.assertEqual(results["feature_c"], 15)  # 0.5 * (10 + 20)


if __name__ == "__main__":
    unittest.main()
