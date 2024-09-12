import numpy as np
from beartype import beartype
from omegaconf import DictConfig, ListConfig

from feature_fabrica.core import Feature
from feature_fabrica.models import PromiseValue
from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import StrOrNumArray
from feature_fabrica.utils import get_logger

logger = get_logger()

class FeatureImporter(Transformation):
    @beartype
    def __init__(self, features: ListConfig[str | DictConfig[str, str]] | None = None, feature: str | None = None, transform_stage: str | None = None):
        """
        Parameters
        ----------
        features : list of str or dict[str, str]
            The names of the features to import. Can be a list of strings, or a dictionary with feature names as
            keys and their specific transform stages as values.
        transform_stage : str, optional
            A single transform stage to apply to all features unless individually specified.
        """
        super().__init__()
        self.features_to_import = []
        if not (features or feature):
            raise ValueError("features or feature should be set.")
        # Deprecate the 'feature' argument
        if feature is not None:
            logger.warning("'feature' is deprecated and will be removed in a future version. Use 'features' instead.")
            self.features_to_import.append((feature, transform_stage))
        else:
            # If features is a list, extract feature names and associated stages
            for feature in features: # type: ignore[union-attr]
                if isinstance(feature, DictConfig):
                    self.features_to_import.append(next(iter(feature.items())))
                else:
                    self.features_to_import.append((feature, transform_stage))

        # Initialize PromiseValues for each feature
        self.data = [PromiseValue() for _ in self.features_to_import]

    @beartype
    def compile(self, features: dict[str, Feature]) -> bool:
        for idx, (feature_name, stage) in enumerate(self.features_to_import):
            if feature_name not in features:
                raise ValueError(f"Feature '{feature_name}' not found.")

            feature_obj = features[feature_name]

            if stage is not None:
                if stage not in feature_obj.transformation:
                    raise ValueError(f"Transform stage '{stage}' not found in feature '{feature_name}'.")
                feature_obj._export_to_features[stage].append(self.data[idx])
            else:
                feature_obj._before_compute_hooks.append(self.data[idx])
        return False

    @beartype
    def execute(self) -> StrOrNumArray:
        if len(self.features_to_import) == 1:
            return self.data[0].value
        else:
            return np.array([promise_value.value for promise_value in self.data])
