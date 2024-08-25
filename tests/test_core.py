# test_core.py
import unittest
from feature_fabrica.core import FeatureSet
from pydantic import ValidationError


class TestFeatureSet(unittest.TestCase):
    def test_load_features(self):
        feature_set = FeatureSet("examples/basic_features.yaml")
        self.assertIn("feature_a", feature_set.features)
        self.assertIn("feature_c", feature_set.features)

    def test_invalid_value_type(self):
        data = {
            "feature_a": "invalid_string",
            "feature_b": 20,
        }  # Expecting float, got str
        feature_set = FeatureSet("examples/basic_features.yaml")
        with self.assertRaises(ValidationError):
            feature_set.compute_all(data)

    def test_compute_all(self):
        data = {"feature_a": 10.0, "feature_b": 20.0}
        feature_set = FeatureSet("examples/basic_features.yaml")
        results = feature_set.compute_all(data)
        self.assertEqual(
            results["feature_c"], 2.772588722239781
        )  # log(0.5 * (10 + 20) + 1)


if __name__ == "__main__":
    unittest.main()
