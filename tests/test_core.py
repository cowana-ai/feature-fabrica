# test_core.py
import unittest

import numpy as np

from feature_fabrica.core import FeatureManager


class TestFeatureSet(unittest.TestCase):
    def test_load_features(self):
        feature_manager = FeatureManager(
            config_path="./examples", config_name="basic_features"
        )
        self.assertIn("feature_a", feature_manager.features)
        self.assertIn("feature_c", feature_manager.features)

    def test_compute_features_single_data(self):
        data = {
            "feature_a": np.array([10], dtype=np.float32),
            "feature_b": np.array([20], dtype=np.float32),
            "feature_e": np.array(["orange"]),
        }
        feature_manager = FeatureManager(
            config_path="./examples", config_name="basic_features"
        )
        results = feature_manager.compute_features(data)
        self.assertEqual(results["feature_c"], 25.0)  # 0.5 * (10 + 20 * 2)
        self.assertEqual(results.feature_c, 25.0)  # 0.5 * (10 + 20 * 2)
        self.assertEqual(
            feature_manager.features.feature_c.feature_value.value, 25.0
        )  # 0.5 * (10 + 20)
        np.testing.assert_array_equal(results.feature_e, np.array([[0, 1]]))

    def test_compute_features_array(self):
        data = {
            "feature_a": np.array(list(range(100)), dtype=np.float32),
            "feature_b": np.array(list(range(100, 200)), dtype=np.float32),
            "feature_e": np.array(["orange"]),
        }
        feature_manager = FeatureManager(
            config_path="./examples", config_name="basic_features"
        )
        results = feature_manager.compute_features(data)
        # Assertions for array results
        expected_values = 0.5 * (
            np.array(data["feature_a"]) + np.array(data["feature_b"]) * 2
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
