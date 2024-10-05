from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from beartype import beartype
from omegaconf import DictConfig, ListConfig

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import AnyArray

if TYPE_CHECKING:
    from feature_fabrica.models import PromiseValue


class FeatureImporter(Transformation):
    _name_ = "import"
    @beartype
    def __init__(self, iterable: ListConfig[str | DictConfig[str, str]] | None = None, feature: str | None = None):
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
        features_to_import = []
        if not (iterable or feature):
            raise ValueError("features or feature should be set.")
        # Deprecate the 'feature' argument
        if feature is not None:
            features_to_import.append(feature)
        else:
            # If features is a list, extract feature names and associated stages
            for feature in iterable: # type: ignore[union-attr]
                if isinstance(feature, DictConfig):
                    key, value = next(iter(feature.items()))
                    features_to_import.append(f"{key}:{value}")
                else:
                    features_to_import.append(feature)
        # Initialize PromiseValues for each feature
        self.iterable: list[PromiseValue | str] = features_to_import

    @beartype
    def execute(self) -> AnyArray | list[AnyArray]:
        if len(self.iterable) == 1:
            return self.iterable[0]._get_value() # type: ignore
        else:
            imported_list = [promise_value._get_value() for promise_value in self.iterable] # type: ignore
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
