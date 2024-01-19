"""Tokenizer that encodes a symbolic trace into a sequence encoding"""

import logging
from typing import List, Type

from ..registry import register_type
from ..tokenizers import SeqToSeqTokenizer
from .symbolic_trace import SymbolicTrace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class SymTraceToSeqTokenizer(SeqToSeqTokenizer):
    def __init__(
        self, notation: str = "infix", dtype: Type[SymbolicTrace] = SymbolicTrace, **kwargs
    ) -> None:
        self.notation = notation
        super().__init__(dtype=dtype, **kwargs)

    def encode_tokens(self, data: SymbolicTrace, **kwargs) -> List[str]:
        return data.to_tokens(notation=self.notation, **kwargs)

    def decode_tokens(self, tokens: List[str], **kwargs) -> SymbolicTrace:
        return SymbolicTrace.from_tokens(tokens=tokens, notation=self.notation, **kwargs)
