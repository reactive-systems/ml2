"""Utility to load dataset"""

import logging

from .dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(name: str, project: str = None, **kwargs) -> Dataset:
    from ..registry import type_from_str

    config = Dataset.fetch_config(name=name, project=project)
    if "type" not in config:
        raise Exception("Dataset type not specified in config")
    dataset_type = type_from_str(config["type"], bound=Dataset)
    return dataset_type.load(name=name, project=project, **kwargs)
