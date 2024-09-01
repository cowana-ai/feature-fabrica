# mypy: ignore-errors
import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from .utils import get_logger

logger = get_logger()


def load_yaml(config_path: str, config_name: str) -> DictConfig:
    # Get the current working directory
    current_working_directory = os.getcwd()
    logger.info(
        f"Reading Feature Definition YAML from {config_path}/{config_name}.yaml"
    )
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)
    feature_specs = compose(
        config_name=config_name,
        overrides=[f"hydra.searchpath=['{current_working_directory}']"],
    )
    return feature_specs
