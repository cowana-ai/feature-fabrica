# core.py
from .models import FeatureSpec, FeatureValue
from .yaml_parser import load_yaml, validate_feature_spec
from .transform import scale_feature


class Feature:
    def __init__(self, name: str, spec: dict):
        self.name = name
        # print(spec)
        self.feature_value = None
        self.spec = FeatureSpec(**spec)
        self.dependencies = self.spec.dependencies
        self.transformation = self.spec.transformation
        self.params = self.spec.params

    def compute(self, value, dependencies=None):
        # If the feature has dependencies, compute them first
        if dependencies:
            dependent_values = [d_fv.feature_value.value for d_fv in dependencies]
        else:
            dependent_values = []

        # Apply the transformation function if specified
        if self.transformation:
            if self.transformation == "scale_feature":
                result = scale_feature(
                    sum(dependent_values), self.params.get("factor", 1)
                )
            else:
                # Handle other transformations if needed
                result = dependent_values[0] if dependent_values else None
        else:
            assert len(dependent_values) == 0
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
                dependencies=[self.features[f_name] for f_name in feature.dependencies],
            )

        return results
