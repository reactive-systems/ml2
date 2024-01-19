"""Utility to load vocabulary"""

from .vocabulary import Vocabulary


def load_vocabulary(name: str, project: str = None, **kwargs) -> Vocabulary:
    from ..registry import type_from_str

    config = Vocabulary.fetch_config(name=name, project=project)
    if "type" not in config:
        raise Exception("Vocabulary type not specified in config")
    vocabulary_type = type_from_str(config["type"], bound=Vocabulary)
    return vocabulary_type.load(name=name, project=project, **kwargs)
