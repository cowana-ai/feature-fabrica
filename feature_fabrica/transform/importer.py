from beartype import beartype

from feature_fabrica.core import Feature
from feature_fabrica.models import PromiseValue
from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import StrOrNumArray


class FeatureImporter(Transformation):
    @beartype
    def __init__(self, feature: str, transform_stage: str | None = None):
        """
        Parameters
        ----------
        feature : str
            The name of the feature to import.
        transform_stage : str, optional
            The specific stage of transformation from which to import data. If not provided (None),
            the initial raw feature value before any transformations will be used.
        """
        super().__init__()
        self.feature = feature
        self.transform_stage = transform_stage
        self.data = PromiseValue()

    @beartype
    def compile(self, features: dict[str, Feature]) -> bool:
        if self.feature not in features:
            raise ValueError(f"Feature '{self.feature}' not found.")

        feature_obj = features[self.feature]
        if self.transform_stage is not None:
            if self.transform_stage not in feature_obj.transformation:
                raise ValueError(f"Transform stage '{self.transform_stage}' not found in feature '{self.feature}'.")
            feature_obj._export_to_features[self.transform_stage].append(self.data)
        else:
            feature_obj._before_compute_hooks.append(self.data)
        return False

    @beartype
    def execute(self) -> StrOrNumArray:
        if self.data.value is None:
            raise ValueError("Data has not been set.")
        return self.data.value
