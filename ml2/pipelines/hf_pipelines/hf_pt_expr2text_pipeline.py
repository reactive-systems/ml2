"""HuggingFace PyTorch expression to text pipeline"""

import logging
from typing import Generic, TypeVar

from ...dtypes import String
from ...registry import register_type
from .hf_pt_text2text_pipeline import HFPTText2TextPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


I = TypeVar("I", bound=String)
T = TypeVar("T", bound=String)


@register_type
class HFPTExpr2TextPipeline(HFPTText2TextPipeline[I, T], Generic[I, T]):
    def __init__(
        self,
        input_notation: str = "infix",
        input_kwargs: dict = None,
        **kwargs,
    ):
        self.input_notation = input_notation
        input_kwargs = input_kwargs if input_kwargs is not None else {}
        input_kwargs["notation"] = input_notation
        super().__init__(input_kwargs=input_kwargs, **kwargs)
