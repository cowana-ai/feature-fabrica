# utils.py
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from beartype import beartype
from loguru import logger
from omegaconf import DictConfig, ListConfig

from feature_fabrica.exceptions import CyclicDependencyError

logger_set = False

@beartype
def setup_logger(
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                      "<level>{message}</level>",
    log_level: str = "DEBUG",
    log_file: str | None = None,
    log_rotation: str | None = None,
    log_retention: str | None = None,
    log_compression: str | None = None,
    colorize_stdout: bool = True
):
    global logger_set

    # Remove the default logger
    logger.remove()

    # Add stdout with formatting and color
    logger.add(sys.stdout, format=log_format, level=log_level, colorize=colorize_stdout)

    # Add file handler if log_file is provided
    if log_file:
        logger.add(
            log_file,
            format=log_format.replace('<level>', '').replace('</level>', ''),
            level=log_level,
            rotation=log_rotation,
            retention=log_retention,
            compression=log_compression,
        )

    logger_set = True


def get_logger(**kwargs):
    if not logger_set:
        setup_logger(**kwargs)
    return logger


def verify_dependencies(dependencies_count: dict[str, int]):
    logger = get_logger()
    if 0 in dependencies_count.values():
        loop_features = [f_name for f_name, c in dependencies_count.items() if c == 0]
        logger.debug(
            f"Cyclic dependency detected! The following features form a cycle: {loop_features}"
        )
        raise CyclicDependencyError(loop_features)

def is_list_like(x: ListConfig | list[Any]) -> bool:
    return isinstance(x, ListConfig) or isinstance(x, list)

def is_dict_like(x: DictConfig | dict[Any, Any]) -> bool:
    return isinstance(x, DictConfig) or isinstance(x, dict)

def compute_all_transformations(transformations: Callable | list[Callable] | dict[str, Callable], initial_value: np.ndarray | None = None, get_intermediate_results: bool = False):
    intermediate_results = []
    if is_dict_like(transformations):
        prev_value = initial_value
        for (
            transformation_name,
            transformation_fn,
        ) in transformations.items(): # type: ignore[union-attr]
            result = transformation_fn(prev_value) if transformation_fn.expects_data else transformation_fn() # type: ignore
            prev_value = result.value

            if get_intermediate_results:
                intermediate_results.append((transformation_name, result))

    elif is_list_like(transformations):
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
