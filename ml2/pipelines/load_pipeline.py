"""Utility to load pipeline"""


import logging

from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pipeline(name: str, project: str = None, **kwargs) -> Pipeline:
    from ..registry import type_from_str

    config = Pipeline.fetch_config(name=name, project=project)
    if "type" not in config:
        raise Exception("Pipeline type not specified in config")
    pipeline_type = type_from_str(config["type"], bound=Pipeline)
    return pipeline_type.load(name=name, project=project, **kwargs)
