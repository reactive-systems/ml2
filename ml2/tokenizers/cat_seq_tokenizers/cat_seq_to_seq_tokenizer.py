"""Tokenizer that encodes a categorized sequence into a sequence encoding"""

from typing import Generic, List, Type, TypeVar

import numpy as np
import tensorflow as tf
import torch

from ...dtypes import Cat, CatSeq, Seq
from ...registry import register_type
from ..cat_tokenizers import CatToIdTokenizer
from ..seq_tokenizers import SeqToSeqTokenizer
from ..to_seq_tokenizer import ToSeqTokenizer

C = TypeVar("C", bound=Cat)
S = TypeVar("S", bound=Seq)
CSP = CatSeq[C, S]
T = TypeVar("T", bound=CSP)


@register_type
class CatSeqToSeqTokenizer(ToSeqTokenizer[T], Generic[T]):
    def __init__(
        self,
        dtype: Type[T],
        swap: bool = False,
        np_int_dtype: type = np.int32,
        pt_int_dtype: torch.dtype = torch.int32,
        tf_int_dtype: tf.DType = tf.int32,
        **kwargs
    ):
        self.swap = swap

        self.cat_tokenizer = CatToIdTokenizer(
            dtype=None,
            np_int_dtype=np_int_dtype,
            pt_int_dtype=pt_int_dtype,
            tf_int_dtype=tf_int_dtype,
        )

        self.seq_tokenizer = SeqToSeqTokenizer(
            dtype=None,
            start=False,
            eos=False,
            pad=None,
            np_int_dtype=np_int_dtype,
            pt_int_dtype=pt_int_dtype,
            tf_int_dtype=tf_int_dtype,
        )

        super().__init__(
            dtype=dtype,
            np_int_dtype=np_int_dtype,
            pt_int_dtype=pt_int_dtype,
            tf_int_dtype=tf_int_dtype,
            **kwargs
        )

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
        return self.dtype.from_cat_seq_tokens(cat_token=cat_token, seq_tokens=seq_tokens, **kwargs)
