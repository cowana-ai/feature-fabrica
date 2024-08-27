# test_core.py
import unittest
from feature_fabrica.core import FeatureManager
from pydantic import ValidationError


class TestFeatureSet(unittest.TestCase):
    def test_load_features(self):
        feature_manager = FeatureManager(
            config_path="../examples", config_name="basic_features"
        )
        self.assertIn("feature_a", feature_manager.features)
        self.assertIn("feature_c", feature_manager.features)

    def test_invalid_value_type(self):
        data = {
            "feature_a": "invalid_string",
            "feature_b": 20,
        }  # Expecting float, got str
        feature_manager = FeatureManager(
            config_path="../examples", config_name="basic_features"
        )
        feature_manager.compile()
        with self.assertRaises(ValidationError):
            feature_manager.compute_all(data)

    def test_compute_all(self):
        data = {"feature_a": 10.0, "feature_b": 20.0}
        feature_manager = FeatureManager(
            config_path="../examples", config_name="basic_features"
        )
        feature_manager.compile()
        results = feature_manager.compute_all(data)
        self.assertEqual(results["feature_c"], 15.0)  # 0.5 * (10 + 20)
        self.assertEqual(results.feature_c, 15.0)  # 0.5 * (10 + 20)
        self.assertEqual(
            feature_manager.features.feature_c.feature_value.value, 15.0
        )  # 0.5 * (10 + 20)


if __name__ == "__main__":
    unittest.main()
