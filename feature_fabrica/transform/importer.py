from beartype import beartype

from feature_fabrica.core import Feature
from feature_fabrica.models import FeatureValue
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
        self.data = None

    @beartype
    def compile(self, features: dict[str, Feature]) -> bool:
        assert self.feature in features
        if self.transform_stage is not None:
            assert self.transform_stage in features[self.feature].transformation
            features[self.feature]._export_to_features[self.transform_stage].append(self)
        else:
            features[self.feature]._before_compute_hooks.append(self.pass_data)
        return False

    @beartype
    def pass_data(self, data: StrOrNumArray) -> None:
        self.data = data

    @beartype
    def execute(self) -> StrOrNumArray:
        if isinstance(self.data, FeatureValue):
            return self.data.value
        return self.data
