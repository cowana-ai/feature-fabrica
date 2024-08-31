# mypy: ignore-errors
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from .utils import get_logger

logger = get_logger()


def load_yaml(config_path: str, config_name: str) -> DictConfig:
    logger.info(
        f"Reading Feature Definition YAML from {config_path}/{config_name}.yaml"
    )
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)
    feature_specs = compose(config_name=config_name)
    return feature_specs
