# test_core.py
import unittest
from feature_fabrica.core import FeatureManager
from pydantic import ValidationError
import numpy as np


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
            "feature_e": "Hellow World",
        }  # Expecting float, got str
        feature_manager = FeatureManager(
            config_path="../examples", config_name="basic_features"
        )
        with self.assertRaises(ValidationError):
            feature_manager.compute_features(data)

    def test_compute_features_single_data(self):
        data = {"feature_a": 10.0, "feature_b": 20.0, "feature_e": "Hellow World"}
        feature_manager = FeatureManager(
            config_path="../examples", config_name="basic_features"
        )
        results = feature_manager.compute_features(data)
        self.assertEqual(results["feature_c"], 15.0)  # 0.5 * (10 + 20)
        self.assertEqual(results.feature_c, 15.0)  # 0.5 * (10 + 20)
        self.assertEqual(
            feature_manager.features.feature_c.feature_value.value, 15.0
        )  # 0.5 * (10 + 20)
        self.assertEqual(results.feature_e, "hellow world")

    def test_compute_features_array(self):
        data = {
            "feature_a": list(range(100)),
            "feature_b": list(range(100, 200)),
            "feature_e": "Hellow World",
        }
        feature_manager = FeatureManager(
            config_path="../examples", config_name="basic_features"
        )
        results = feature_manager.compute_features(data)
        # Assertions for array results
        expected_values = 0.5 * (
            np.array(data["feature_a"]) + np.array(data["feature_b"])
        )

        # Assert that the result is an array
        self.assertIsInstance(results["feature_c"], np.ndarray)
        self.assertIsInstance(results.feature_c, np.ndarray)
        self.assertIsInstance(
            feature_manager.features.feature_c.feature_value.value, np.ndarray
        )

        # Assert that the array has the expected values
        np.testing.assert_array_equal(results["feature_c"], expected_values)
        np.testing.assert_array_equal(results.feature_c, expected_values)
        np.testing.assert_array_equal(
            feature_manager.features.feature_c.feature_value.value, expected_values
        )


if __name__ == "__main__":
    unittest.main()
