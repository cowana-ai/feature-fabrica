# test_core.py
import unittest

import numpy as np

from feature_fabrica.core import FeatureManager


class TestFeatureSet(unittest.TestCase):
    def test_load_features(self):
        feature_manager = FeatureManager(
            config_path="./examples", config_name="basic_features", log_transformation_chain=False
        )
        self.assertIn("feature_a", feature_manager.features)
        self.assertIn("feature_c", feature_manager.features)

    def test_compute_features_single_data(self):
        data = {
            "feature_a": np.array([10], dtype=np.int32),
            "feature_b": np.array([20], dtype=np.int32),
            "feature_e": np.array(["orange"]),
            "feature_f": np.array(["orange "]),
        }
        feature_manager = FeatureManager(
            config_path="./examples", config_name="basic_features", log_transformation_chain=False
        )
        results = feature_manager.compute_features(data)
        self.assertEqual(results["feature_c"], 25.0)  # 0.5 * (10 + 20 * 2)
        self.assertEqual(results.feature_c, 25.0)  # 0.5 * (10 + 20 * 2)
        self.assertEqual(
            feature_manager.features.feature_c.feature_value.value, 25.0
        )  # 0.5 * (10 + 20)
        np.testing.assert_array_equal(results.feature_e, np.array([[0, 1]]))
        self.assertEqual(results["feature_f"], "orange")  # 0.5 * (10 + 20 * 2)

    def test_compute_features_array(self):
        data = {
            "feature_a": np.array(list(range(100)), dtype=np.int32),
            "feature_b": np.array(list(range(100, 200)), dtype=np.int32),
            "feature_e": np.array(["Orange", "Apple"]),
            "feature_f": np.array(["orange "]),
        }
        feature_manager = FeatureManager(
            config_path="./examples", config_name="basic_features", log_transformation_chain=False
        )
        results = feature_manager.compute_features(data)
        # Assertions for array results
        expected_feature_c = 0.5 * (
            np.array(data["feature_a"]) + np.array(data["feature_b"]) * 2
        )
        expected_feature_f = np.array(["orange"])
        expected_feature_e_original = np.array(["Orange", "Apple"])
        expected_feature_e_lower = np.array(["orange", "apple"])
        expected_feature_e_upper = np.array(["ORANGE", "APPLE"])
        expected_feature_e = np.array([[0, 1], [1, 0]], dtype=np.int32)
        expected_feature_e_upper_lower_original = np.array(["ORANGEorangeOrange", "APPLEappleApple"])

        expected_feeture_abd = data["feature_a"] + data["feature_b"] + results["feature_d"]
        # Assert feature c
        self.assertIsInstance(results["feature_c"], np.ndarray)
        self.assertIsInstance(results.feature_c, np.ndarray)
        self.assertIsInstance(
            feature_manager.features.feature_c.feature_value.value, np.ndarray
        )
        np.testing.assert_array_equal(results["feature_c"], expected_feature_c)
        np.testing.assert_array_equal(results.feature_c, expected_feature_c)

        # Assert feature_f
        np.testing.assert_array_equal(results["feature_f"], expected_feature_f)

        # Assert feature_e's
        np.testing.assert_array_equal(results["feature_e_original"], expected_feature_e_original)
        np.testing.assert_array_equal(results["feature_e_lower"], expected_feature_e_lower)
        np.testing.assert_array_equal(results["feature_e_upper"], expected_feature_e_upper)
        np.testing.assert_array_equal(results["feature_e"], expected_feature_e)
        np.testing.assert_array_equal(results["feature_e"], results["feature_e_one_hot"])
        np.testing.assert_array_equal(results["feature_e_upper_lower_original"], expected_feature_e_upper_lower_original)
        np.testing.assert_array_equal(results["feature_abd"], expected_feeture_abd)

        # Assert FeatureValue
        np.testing.assert_array_equal(
            feature_manager.features.feature_c.feature_value.value, expected_feature_c
        )
        np.testing.assert_array_equal(feature_manager.features.feature_c.feature_value[[25, 30]], expected_feature_c[[25, 30]])


if __name__ == "__main__":
    unittest.main()
