# utils.py
import sys

from loguru import logger

from feature_fabrica.exceptions import CyclicDependencyError

logger_set = False


def setup_logger():
    global logger_set
    # TODO: configurable
    # Configuration settings
    LOG_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    LOG_LEVEL = "DEBUG"
    # LOG_FILE = "app.log"
    # LOG_ROTATION = "1 week"  # Log rotation interval
    # LOG_RETENTION = "10 days"  # Log retention duration
    # LOG_COMPRESSION = "zip"  # Compression type for rotated logs

    # Remove the default logger
    logger.remove()

    # Add stdout with formatting and color
    logger.add(sys.stdout, format=LOG_FORMAT, level=LOG_LEVEL, colorize=True)

    # Add file handler without color (colors don't make sense in a file)
    # logger.add(LOG_FILE, format=LOG_FORMAT.replace('<level>', '').replace('</level>', ''),
    #           level=LOG_LEVEL, rotation=LOG_ROTATION, retention=LOG_RETENTION, compression=LOG_COMPRESSION)
    logger_set = True


def get_logger():
    if not logger_set:
        setup_logger()
    return logger


def verify_dependencies(dependencies_count: dict[str, int]):
    logger = get_logger()
    if 0 in dependencies_count.values():
        loop_features = [f_name for f_name, c in dependencies_count.items() if c == 0]
        logger.debug(
            f"Cyclic dependency detected! The following features form a cycle: {loop_features}"
        )
        raise CyclicDependencyError(loop_features)
