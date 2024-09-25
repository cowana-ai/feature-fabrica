# utils.py
import sys

from beartype import beartype
from loguru import logger

from feature_fabrica._internal.instantiate._instantiate import instantiate
from feature_fabrica.exceptions import CyclicDependencyError

# Instantiation related symbols
instantiate = instantiate

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
