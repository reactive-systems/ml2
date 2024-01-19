"""Tokenizer that encodes a categorical data type into an index"""

from typing import Generic, TypeVar

from ...dtypes import Cat
from ...registry import register_type
from ..to_id_tokenizer import ToIdTokenizer

T = TypeVar("T", bound=Cat)


@register_type
class CatToIdTokenizer(ToIdTokenizer[T], Generic[T]):
    def encode_token(self, data: T, **kwargs) -> str:
        return data.token(**kwargs)

    def decode_token(self, token: str, **kwargs) -> T:
        return self.dtype.from_token(token=token, **kwargs)
