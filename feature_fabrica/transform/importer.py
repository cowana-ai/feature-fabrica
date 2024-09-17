import numpy as np
from beartype import beartype
from omegaconf import DictConfig, ListConfig

from feature_fabrica.models import PromiseValue
from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import StrOrNumArray
from feature_fabrica.utils import get_logger, get_promise_manager

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
        promise_manager = get_promise_manager()
        # Initialize PromiseValues for each feature
        self.data: list[PromiseValue] = [promise_manager.get_promise_value(feature, transform_stage) for (feature, transform_stage) in self.features_to_import]

    @beartype
    def execute(self) -> StrOrNumArray | list[StrOrNumArray]:
        if len(self.features_to_import) == 1:
            return self.data[0].value
        else:
            imported_list = [promise_value.value for promise_value in self.data]
            # Determine whether we have mixed types and what the final type should be
            has_float = 0
            has_int = 0
            has_str = 0

            for d in imported_list:
                # mixed type
                if has_float and has_int and has_str:
                    break
                dtype = d.dtype # type: ignore[union-attr]
                if np.issubdtype(dtype, np.floating):
                    has_float += 1
                elif np.issubdtype(dtype, np.integer):
                    has_int += 1
                elif np.issubdtype(dtype, np.str_):
                    has_str += 1
            # Handle mixed numeric types (if both int and float, cast to float)
            if (has_float or has_int) and not has_str:
                return np.array(imported_list, dtype=float if has_float else int)

            # Handle string arrays
            if has_str == len(imported_list):
                return np.array(imported_list, dtype=str)
            # Fallback in case data types are mixed or need custom handling
            return imported_list
