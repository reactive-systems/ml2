"""Tokenizer that encodes a sequence into a sequence encoding"""


from typing import Generic, List, TypeVar

from ...dtypes import Seq
from ...registry import register_type
from ..to_seq_tokenizer import ToSeqTokenizer

T = TypeVar("T", bound=Seq)


@register_type
class SeqToSeqTokenizer(ToSeqTokenizer[T], Generic[T]):
    def encode_tokens(self, data: T, **kwargs) -> List[str]:
        return data.to_tokens(**kwargs)

    def decode_tokens(self, tokens: List[str], **kwargs) -> T:
        return self.dtype.from_tokens(tokens, **kwargs)
