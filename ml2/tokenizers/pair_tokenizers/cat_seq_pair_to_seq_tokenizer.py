"""Tokenizer that encodes a pair of category and sequence into a sequence encoding"""

from typing import Generic, List, TypeVar

from ...dtypes import Cat, Pair, Seq
from ...registry import register_type
from ..cat_tokenizers import CatToIdTokenizer
from ..seq_tokenizers import SeqToSeqTokenizer
from ..to_seq_tokenizer import ToSeqTokenizer

C = TypeVar("C", bound=Cat)
S = TypeVar("S", bound=Seq)
P = Pair[C, S]
T = TypeVar("T", bound=P)


@register_type
class CatSeqPairToSeqTokenizer(ToSeqTokenizer[T], Generic[C, S, T]):
    def __init__(
        self,
        cat_tokenizer: CatToIdTokenizer[C],
        seq_tokenizer: SeqToSeqTokenizer[S],
        swap: bool = False,
        **kwargs
    ):
        self.cat_tokenizer = cat_tokenizer
        self.seq_tokenizer = seq_tokenizer
        self.swap = swap

        super().__init__(**kwargs)

    def encode_tokens(self, data: T, **kwargs) -> List[str]:
        tokens_fst = [self.cat_tokenizer.encode_token(data.cat, **kwargs)]
        tokens_snd = self.seq_tokenizer.encode_tokens(data.seq, **kwargs)
        if self.swap:
            return tokens_snd + tokens_fst
        else:
            return tokens_fst + tokens_snd

    def decode_tokens(self, ids: List[str], **kwargs) -> T:
        if self.swap:
            cat_token = ids[-1]
            seq_tokens = ids[:-1]
        else:
            cat_token = ids[0]
            seq_tokens = ids[1:]
        fst = self.cat_tokenizer.decode_token(cat_token, **kwargs)
        snd = self.seq_tokenizer.decode_tokens(seq_tokens, **kwargs)
        return self.dtype.from_components(fst=fst, snd=snd, **kwargs)
