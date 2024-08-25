# core.py
from .models import FeatureSpec, FeatureValue
from .yaml_parser import load_yaml, validate_feature_spec
from collections.abc import Iterable
import inspect


class Feature:
    def __init__(self, name: str, spec: dict):
        self.name = name
        # print(spec)
        self.feature_value = None
        self.spec = FeatureSpec(**spec)
        self.dependencies = self.spec.dependencies
        self.transformation = self.spec.transformation

    def compute(self, value, dependencies=None):
        # Apply the transformation function if specified
        if self.transformation:
            result = value
            for key, params in self.transformation.items():
                fn = eval(key)
                fn_input = {}
                # Get the function signature
                signature = inspect.signature(fn)
                if "data" in signature.parameters:
                    fn_input["data"] = result
                if params:
                    for fn_arg, fn_val in params.items():
                        if isinstance(fn_val, Iterable):
                            fn_input[fn_arg] = [
                                dependencies[x].feature_value.value
                                if x in dependencies.keys()
                                else x
                                for x in fn_val
                            ]
                        else:
                            fn_input[fn_arg] = (
                                dependencies[fn_val].feature_value.value
                                if fn_val in dependencies.keys()
                                else fn_val
                            )
                result = fn(**fn_input)

        else:
            assert dependencies is None
            result = value

        # Validate the final result with FeatureValue
        self.feature_value = FeatureValue(value=result, data_type=self.spec.data_type)
        return self.feature_value.value


class FeatureSet:
    def __init__(self, yaml_file):
        self.feature_specs = load_yaml(yaml_file)

        self.independent_features = []
        self.dependent_features = []

        self.features = self._build_features()

    def _build_features(self):
        features = {}
        for name, spec in self.feature_specs.items():
            validate_feature_spec(spec)
            feature = Feature(name, spec)

            if not feature.dependencies:
                self.independent_features.append(feature)
            else:
                self.dependent_features.append(feature)

            features[name] = feature

        return features

    def compute_all(self, data):
        results = {}

        for feature in self.independent_features:
            name = feature.name
            value = data[name]
            results[name] = feature.compute(value)

        for feature in self.dependent_features:
            name = feature.name
            results[name] = feature.compute(
                0,
                dependencies={
                    f_name: self.features[f_name] for f_name in feature.dependencies
                },
            )

        return results
