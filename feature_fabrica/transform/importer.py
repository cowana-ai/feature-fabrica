from beartype import beartype

from feature_fabrica.models import FeatureValue
from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import StrOrNumArray


class FeatureImporter(Transformation):
    @beartype
    def __init__(self, feature: str, transform_stage: str|None=None):
        super().__init__()
        self.feature = feature
        self.transform_stage = transform_stage
    @beartype
    def execute(self) -> StrOrNumArray:
        assert isinstance(self.feature, FeatureValue)
        return self.feature.value
