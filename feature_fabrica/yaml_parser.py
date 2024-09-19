# mypy: ignore-errors
import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from feature_fabrica.utils import get_logger

logger = get_logger()


def load_yaml(config_path: str, config_name: str) -> DictConfig:
    # Get the current working directory
    current_working_directory = os.getcwd()
    logger.info(
        f"Reading Feature Definition YAML from {config_path}/{config_name}.yaml"
    )
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=os.path.join(current_working_directory, config_path),
        version_base="1.3",
    )
    feature_specs = compose(
        config_name=config_name,
    )
    return feature_specs
