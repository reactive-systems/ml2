"""Utility to load artifact"""

from typing import Type, Union

from .artifact import Artifact
from .configurable import Configurable
from .registry import type_from_str


def get_artifact_type(config: Union[str, dict]) -> Type:
    if isinstance(config, str):
        config = Artifact.fetch_config(config)
    if "type" in config:
        config_type = config["type"]
    elif "base" in config:
        base_config = Artifact.fetch_config(config["base"])
        config_type = base_config["type"]
    else:
        raise Exception("Could not determine artifact type")

    if isinstance(config_type, str):
        config_type = type_from_str(config_type, bound=Configurable)

    return config_type


def load_artifact(config: Union[str, dict], **kwargs) -> Artifact:
    config_type = get_artifact_type(config)
    return config_type.from_config(config, **kwargs)
