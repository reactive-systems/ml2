"""Tokenizer that encodes an expression into a sequence encoding"""

from typing import Generic, List, TypeVar

from ...dtypes import BinaryExpr
from ...registry import register_type
from ..to_seq_tokenizer import ToSeqTokenizer

T = TypeVar("T", bound=BinaryExpr)


@register_type
class ExprToSeqTokenizer(ToSeqTokenizer[T], Generic[T]):
    def __init__(self, notation: str = "infix", **kwargs):
        self.notation = notation
        super().__init__(**kwargs)

    def encode_tokens(self, data: T, **kwargs) -> List[str]:
        return data.to_tokens(notation=self.notation, **kwargs)

    def decode_tokens(self, tokens: List[str], **kwargs) -> T:
        return self.dtype.from_tokens(tokens, notation=self.notation, **kwargs)

    def vocabulary_filename(self) -> str:
        return super().vocabulary_filename() + "-" + self.notation.value
