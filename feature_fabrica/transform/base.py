from abc import ABC
from collections.abc import Iterable, Mapping
import inspect
from ..core import Feature
import time
from easydict import EasyDict as edict
import numpy as np
from ..utils import get_logger

logger = get_logger()


class Transformation(ABC):
    def __init__(self) -> None:
        self.expects_data = False
        self.v_execute = None

    def compile(self, features: dict[str, Feature] | None) -> bool:
        if features is not None:
            for attr_name, attr_value in self.__dict__.items():
                if attr_name == "expects_data":
                    continue

                if isinstance(attr_value, str) and attr_value in features:
                    setattr(self, attr_name, features[attr_value].feature_value.value)  # type: ignore[attr-defined]
                elif isinstance(attr_value, Iterable):
                    setattr(
                        self,
                        attr_name,
                        [
                            features[item].feature_value.value  # type: ignore[attr-defined]
                            if isinstance(item, str) and item in features
                            else item
                            for item in attr_value
                        ],
                    )
                elif isinstance(attr_value, Mapping):
                    setattr(
                        self,
                        attr_name,
                        {
                            key: features[val].feature_value.value  # type: ignore[attr-defined]
                            if isinstance(val, str) and val in features
                            else val
                            for key, val in attr_value.items()
                        },
                    )
        # Check the signature of the execute method
        execute_signature = inspect.signature(self.execute)
        execute_params = execute_signature.parameters

        try:
            # TODO: polars? numba?
            self.v_execute = np.vectorize(self.execute)
        except Exception as e:
            logger.warning(
                f"Warning: Could not np.vectorize {type(self).__name__} due to {e}, operations will be executed as usual"
            )

        # Raise an error if execute expects more than one argument (excluding 'self')
        if len(execute_params) > 2:  # 'self' and one additional argument
            raise TypeError(
                f"{self.__class__.__name__}.execute expects too many arguments. "
                f"Expected 1 argument, but got {len(execute_params) - 1}."
            )
        assert hasattr(self, "expects_data")

        self.expects_data = len(execute_params) == 1
        return self.expects_data

    def execute(self, *args):
        raise NotImplementedError()

    @logger.catch
    def __call__(self, *args):
        # Start time
        start_time = time.time()
        if self.v_execute is not None:
            value = self.v_execute(*args)
        else:
            if len(args) == 1:
                input = args[0]
                # treat str as single input
                if isinstance(input, Iterable) and not isinstance(input, str):
                    value = np.stack([self.execute(v) for v in input])
                else:
                    value = self.execute(input)

            else:
                value = self.execute()
        # End time
        end_time = time.time()

        return edict(
            start_time=start_time,
            value=value,
            end_time=end_time,
        )
