"""Sequence to sequence pipeline"""

import logging
from typing import Generic, Optional, TypeVar

from ..dtypes import DType
from ..registry import register_type
from ..tokenizers import EOS_TOKEN, PAD_TOKEN, START_TOKEN, ToSeqTokenizer
from .sl_pipeline import SLPipeline

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class Seq2SeqPipeline(SLPipeline[I, T], Generic[I, T]):
    def __init__(
        self,
        input_tokenizer: ToSeqTokenizer[I],
        target_tokenizer: ToSeqTokenizer[T],
        max_input_length: int,
        max_target_length: int,
        **kwargs,
    ) -> None:
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        super().__init__(
            input_tokenizer=input_tokenizer, target_tokenizer=target_tokenizer, **kwargs
        )

    @property
    def input_pad_id(self) -> Optional[int]:
        return self.input_tokenizer.vocabulary.token_to_id.get(PAD_TOKEN, None)

    @property
    def input_eos_id(self) -> Optional[int]:
        return self.input_tokenizer.vocabulary.token_to_id.get(EOS_TOKEN, None)

    @property
    def target_start_id(self) -> Optional[int]:
        return self.target_tokenizer.vocabulary.token_to_id.get(START_TOKEN, None)

    @property
    def target_eos_id(self) -> Optional[int]:
        return self.target_tokenizer.vocabulary.token_to_id.get(EOS_TOKEN, None)

    @property
    def target_pad_id(self) -> Optional[int]:
        return self.target_tokenizer.vocabulary.token_to_id.get(PAD_TOKEN, None)
