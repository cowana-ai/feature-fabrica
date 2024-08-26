# core.py
from .models import FeatureSpec, FeatureValue
from .yaml_parser import load_yaml, validate_feature_spec
from collections import defaultdict
from omegaconf import OmegaConf
from hydra.utils import instantiate


class Feature:
    def __init__(self, name: str, spec: OmegaConf):
        self.name = name
        # print(spec)
        self.feature_value = None
        self.spec = FeatureSpec(**spec)
        self.dependencies = self.spec.dependencies
        self.transformation = instantiate(self.spec.transformation)

    def compute(self, value, dependencies=None):
        # Apply the transformation function if specified
        if self.transformation:
            result = value
            for key, transformation_obj in self.transformation.items():
                expects_data = transformation_obj.compile(dependencies)
                if expects_data:
                    result = transformation_obj.execute(result)
                else:
                    result = transformation_obj.execute()

        else:
            assert dependencies is None
            result = value

        # Validate the final result with FeatureValue
        self.feature_value = FeatureValue(value=result, data_type=self.spec.data_type)
        return self.feature_value.value


class FeatureSet:
    def __init__(self, config_path: str, config_name: str):
        self.feature_specs: OmegaConf = load_yaml(
            config_path=config_path, config_name=config_name
        )

        self.independent_features: list[str] = []
        self.dependent_features: list[str] = []

        self.features = self._build_features()
        self.queue: list[str] = []

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

    def compile(self):
        visited = defaultdict(int)
        for feature in self.independent_features:
            self.queue.append(feature)
            visited[feature.name] = 1
        dependent_features_sorted = sorted(
            self.dependent_features,
            key=lambda f: sum(visited[n] for n in f.dependencies),
            reverse=True,
        )
        for feature in dependent_features_sorted:
            self.queue.append(feature)
            visited[feature.name] = 1

    def compute_all(self, data):
        results = {}

        assert len(self.queue) == len(self.features)
        for feature in self.queue:
            name = feature.name
            is_independent = not feature.dependencies
            results[name] = feature.compute(
                value=data[name] if is_independent else 0,
                dependencies=None
                if is_independent
                else {f_name: self.features[f_name] for f_name in feature.dependencies},
            )

        return results
