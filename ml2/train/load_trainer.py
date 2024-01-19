"""Utility to load trainer"""

import logging

from .trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trainer(name: str, project: str = None, **kwargs) -> Trainer:
    from ..registry import type_from_str

    config = Trainer.fetch_config(name=name, project=project)
    if "type" not in config:
        raise Exception("Trainer type not specified in config")
    trainer_type = type_from_str(config["type"], bound=Trainer)
    return trainer_type.load(name=name, project=project, **kwargs)
