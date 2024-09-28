# utils.py
from collections.abc import Callable

import numpy as np
from omegaconf import OmegaConf


def compute_all_transformations(
    transformations: Callable | list[Callable] | dict[str, Callable],
    initial_value: np.ndarray | None = None,
    get_intermediate_results: bool = False
):
    # Create a sequence of transformation functions
    if OmegaConf.is_dict(transformations):
        transformation_sequence = transformations.items() # type: ignore
    elif OmegaConf.is_list(transformations):
        transformation_sequence = enumerate(transformations) # type: ignore
    else:
        # Handle the single transformation case
        transformation_sequence = [(None, transformations)] # type: ignore

    # Initialize the previous value and the list to store intermediate results
    prev_value = initial_value
    intermediate_results = []

    # Process each transformation in the sequence
    for idx, transformation_fn in transformation_sequence:
        # Execute the transformation based on whether it expects data
        result = transformation_fn(prev_value) if transformation_fn.expects_data else transformation_fn() # type: ignore

        # Update the previous value with the result's value
        prev_value = result.value

        # Collect intermediate results if requested
        if get_intermediate_results:
            intermediate_results.append((idx, result) if idx is not None else (transformation_fn.__name__, result))

    # Return the final result, along with intermediate results if requested
    return (result, intermediate_results) if get_intermediate_results else result


def compile_all_transformations(transformations: Callable | list[Callable] | dict[str, Callable], dependencies):
    # Create a sequence of transformation functions
    if OmegaConf.is_dict(transformations):
        transformation_sequence = transformations.values() # type: ignore
    elif OmegaConf.is_list(transformations):
        transformation_sequence = transformations # type: ignore
    else:
        # Handle the single transformation case
        transformation_sequence = [transformations] # type: ignore

    # Process each transformation in the sequence
    for transformation_fn in transformation_sequence:
        transformation_fn.compile(dependencies) # type: ignore
