from beartype import beartype

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import AnyArrya, is_valid_numpy_dtype


class AsType(Transformation):
    @beartype
    def __init__(self, dtype: str):
        if not is_valid_numpy_dtype(dtype):
            raise ValueError(f"dtype = {dtype} is not valid numpy data type!")

        self.dtype = dtype
    @beartype
    def execute(self, data: AnyArrya) -> AnyArrya:
        return data.astype(self.dtype)
