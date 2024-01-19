"""Tokenizer that encodes into a sequence encoding with tree positional encoding"""

import logging
from typing import Any, Dict, Generic, TypeVar

from ..dtypes import DType, TPEFormat
from ..registry import register_type
from .to_seq_pos_tokenizer import ToSeqPosTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=DType)


@register_type
class ToSeqTPETokenizer(ToSeqPosTokenizer[T], Generic[T]):
    def __init__(self, tpe_format: TPEFormat = TPEFormat.BRANCHUP, **kwargs) -> None:
        self.tpe_format = tpe_format
        super().__init__(**kwargs)

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_tpe_pad(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "tpe_pad" not in config:
                config["tpe_pad"] = config.pop("pos_pad")
                annotations["tpe_pad"] = annotations.pop("pos_pad")

        return super().config_postprocessors() + [postprocess_tpe_pad]

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_tpe_pad(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "tpe_pad" in config:
                config["pos_pad"] = config.pop("tpe_pad")

        return [preprocess_tpe_pad] + super().config_preprocessors()
