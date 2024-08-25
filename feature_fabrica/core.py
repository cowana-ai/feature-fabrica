# core.py
from .yaml_parser import load_yaml, validate_feature_spec
from .transform import scale_feature


class Feature:
    def __init__(self, name, spec):
        self.name = name
        self.spec = spec
        self.dependencies = spec.get("dependencies", [])
        self.transformation = spec.get("transformation", None)
        self.params = spec.get("params", {})

    def compute(self, data):
        # If the feature has dependencies, compute them first
        if self.dependencies:
            dependent_values = [data[dep] for dep in self.dependencies]
        else:
            dependent_values = []

        # Apply the transformation function if specified
        if self.transformation:
            if self.transformation == "scale_feature":
                return scale_feature(
                    sum(dependent_values), self.params.get("factor", 1)
                )
            # Add more transformations as needed
        else:
            # If no transformation, just return the first dependency's value
            return dependent_values[0] if dependent_values else None


class FeatureSet:
    def __init__(self, yaml_file):
        self.feature_specs = load_yaml(yaml_file)
        self.features = self._build_features()

    def _build_features(self):
        features = {}
        for name, spec in self.feature_specs.items():
            validate_feature_spec(spec)
            features[name] = Feature(name, spec)
        return features

    def compute_all(self, data):
        results = {}
        for name, feature in self.features.items():
            results[name] = feature.compute(data)
        return results
