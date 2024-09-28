# test_core.py
import unittest

import numpy as np

from feature_fabrica.core import FeatureManager
from feature_fabrica.transform import Transformation


class MyCustomTransform(Transformation):
    _name_ = "my_custom_transform"
    def execute(self, data):
        return data * 2

class TestCustomTransform(unittest.TestCase):
    def test_custom_transform(self):
        data = {
        "feature_a": np.array([10, 20], dtype=np.int32)
        }

        feature_manager = FeatureManager(config_path="./examples", config_name="custom_transform")
        results = feature_manager.compute_features(data)
        np.testing.assert_array_equal(results.feature_a, data["feature_a"] * 2)
