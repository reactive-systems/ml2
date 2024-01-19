"""Tokenizer that encodes a pair into a sequence encoding"""

from typing import Generic, List, TypeVar

from ...dtypes import DType, Pair
from ...registry import register_type
from ..to_seq_pos_tokenizer import ToSeqTokenizer

F = TypeVar("F", bound=DType)
S = TypeVar("S", bound=DType)
P = Pair[F, S]
T = TypeVar("T", bound=P)


@register_type
class PairToSeqTokenizer(ToSeqTokenizer[T], Generic[F, S, T]):
    def __init__(
        self,
        tokenizer_fst: ToSeqTokenizer[F],
        tokenizer_snd: ToSeqTokenizer[S],
        delimiter_token: str = None,
        swap: bool = False,
        **kwargs
    ):
        self.tokenizer_fst = tokenizer_fst
        self.tokenizer_snd = tokenizer_snd

        self.delimiter_token = delimiter_token
        self.swap = swap

        super().__init__(**kwargs)

    def encode_tokens(self, data: T, **kwargs) -> List[str]:
        tokens_fst = self.tokenizer_fst.encode_tokens(data.fst, **kwargs)
        tokens_snd = self.tokenizer_snd.encode_tokens(data.snd, **kwargs)
        delimiter = [] if self.delimiter_token is None else [self.delimiter_token]
        if self.swap:
            return tokens_snd + delimiter + tokens_fst
        else:
            return tokens_fst + delimiter + tokens_snd

    def decode_tokens(self, ids: List[str], **kwargs) -> T:
        if self.delimiter_token is None:
            raise Exception("Can not decode tokens without delimiter token")
        delimiter_count = ids.count(self.delimiter_token)
        if delimiter_count == 0:
            raise Exception("No delimiter token")
        elif delimiter_count > 1:
            raise Exception("More than one delimiter token")
        else:
            delimiter_index = ids.index(self.delimiter_token)
            tokens_fst = ids[:delimiter_index]
            tokens_snd = ids[delimiter_index + 1 :]
            if self.swap:
                fst = self.tokenizer_fst.decode_tokens(tokens_snd)
                snd = self.tokenizer_snd.decode_tokens(tokens_fst)
            else:
                fst = self.tokenizer_fst.decode_tokens(tokens_fst)
                snd = self.tokenizer_snd.decode_tokens(tokens_snd)
            return self.dtype.from_components(fst=fst, snd=snd, **kwargs)
