"""Utility to load verifier"""

import logging

from .verifier import Verifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_verifier_from_general_config(config, log_name: str = "verifier") -> Verifier:
    from ..registry import type_from_str

    if isinstance(config, str):
        verifier = type_from_str(config, bound=Verifier)()

    elif isinstance(config, dict):
        if "type" not in config:
            raise Exception(f"Type not specified in {log_name} config {config}")
        verifier_type = type_from_str(config.pop("type"), bound=Verifier)
        verifier = verifier_type.from_config(config)

    else:
        raise Exception(f"Invalid {log_name} config {config}")
    return verifier
