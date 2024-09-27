# utils.py
from collections.abc import Callable

import numpy as np
from omegaconf import OmegaConf


def compute_all_transformations(transformations: Callable | list[Callable] | dict[str, Callable],
                                initial_value: np.ndarray | None = None,
                                get_intermediate_results: bool = False):
    intermediate_results = []
    if OmegaConf.is_dict(transformations):
        prev_value = initial_value
        for (
            transformation_name,
            transformation_fn,
        ) in transformations.items(): # type: ignore[union-attr]
            result = transformation_fn(prev_value) if transformation_fn.expects_data else transformation_fn() # type: ignore
            prev_value = result.value

            if get_intermediate_results:
                intermediate_results.append((transformation_name, result))

    elif OmegaConf.is_list(transformations):
        prev_value = initial_value
        for (
            idx,
            transformation_fn,
        ) in enumerate(transformations): # type: ignore
            result = transformation_fn(prev_value) if transformation_fn.expects_data else transformation_fn()  # type: ignore
            prev_value = result.value

            if get_intermediate_results:
                intermediate_results.append((idx, result))
    else:
        result = transformations(initial_value) if transformations.expects_data else transformations()  # type: ignore

    return (result, intermediate_results) if get_intermediate_results else result

def compile_all_transformations(transformations: Callable | list[Callable] | dict[str, Callable], dependencies):
    if OmegaConf.is_dict(transformations):
        for transformation_fn in transformations.values(): # type: ignore[union-attr]
            transformation_fn.compile(dependencies) # type: ignore

    elif OmegaConf.is_list(transformations):
        for transformation_fn in transformations: # type: ignore
            transformation_fn.compile(dependencies) # type: ignore
    else:
        transformations.compile(dependencies)  # type: ignore
