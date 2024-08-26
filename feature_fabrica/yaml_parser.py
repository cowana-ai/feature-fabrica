# mypy: ignore-errors
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra


def load_yaml(config_path: str, config_name: str) -> OmegaConf:
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)
    feature_specs = compose(config_name=config_name)
    return feature_specs


def validate_feature_spec(spec):
    # Add validation logic here (e.g., check required fields)
    pass
