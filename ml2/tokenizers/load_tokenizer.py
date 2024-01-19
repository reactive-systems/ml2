"""Utility to load tokenizer"""

import logging

from ..tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenizer(name: str, project: str = None, **kwargs) -> Tokenizer:
    from ..registry import type_from_str

    config = Tokenizer.fetch_config(name=name, project=project)
    if "type" not in config:
        raise Exception("Tokenizer type not specified in config")
    tokenizer_type = type_from_str(config["type"], bound=Tokenizer)
    return tokenizer_type.load(name=name, project=project, **kwargs)
