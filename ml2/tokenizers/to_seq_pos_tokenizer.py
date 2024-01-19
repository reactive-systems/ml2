"""Tokenizer that encodes to a sequence encoding with positional encoding"""

import copy
from abc import abstractmethod
from typing import Generic, List, Tuple, TypeVar

import tensorflow as tf

from ..dtypes import DType
from ..registry import register_type
from .to_seq_tokenizer import SeqEncoding, ToSeqTokenizer
from .tokenizer import TFEncoding, TokenizationException

T = TypeVar("T", bound=DType)


class SeqPosEncoding(SeqEncoding):
    def __init__(
        self,
        tokens: List[str],
        pad_tokens: List[str],
        ids: List[int],
        pos_enc: List[List[int]],
        pad_pos_enc: List[List[int]],
    ) -> None:
        self.pos_enc = pos_enc
        self.pad_pos_enc = pad_pos_enc
        super().__init__(
            tokens=tokens,
            pad_tokens=pad_tokens,
            ids=ids,
        )


@register_type
class ToSeqPosTokenizer(ToSeqTokenizer[T], Generic[T]):
    def __init__(self, pos_pad: int = None, **kwargs):
        self.pos_pad = pos_pad
        super().__init__(**kwargs)

    @property
    def tf_signature(self) -> Tuple[tf.TypeSpec, tf.TypeSpec]:
        seq_spec = tf.TensorSpec(shape=(self.pad,), dtype=self.tf_int_dtype)
        pe_spec = tf.TensorSpec(shape=(self.pad, self.pos_pad), dtype=self.tf_int_dtype)
        return (seq_spec, pe_spec)

    def encode(self, data: T, **kwargs) -> SeqPosEncoding:
        seq_enc = super().encode(data=data, **kwargs)
        pos_enc = self.encode_pos_enc(data, **kwargs)
        assert len(pos_enc) == len(seq_enc.tokens)
        pad_pos_enc = self.pad_pos_enc(pos_enc=pos_enc)
        return SeqPosEncoding(
            tokens=seq_enc.tokens,
            pad_tokens=seq_enc.pad_tokens,
            ids=seq_enc.ids,
            pos_enc=pos_enc,
            pad_pos_enc=pad_pos_enc,
        )

    @abstractmethod
    def encode_pos_enc(self, data: T, **kwargs) -> List[List[int]]:
        raise NotImplementedError()

    def encode_tf(self, data: T, **kwargs) -> TFEncoding:
        encoding = self.encode(data, **kwargs)
        ids_tensor = tf.constant(encoding.ids, dtype=self.tf_int_dtype)
        ppe_tensor = tf.constant(encoding.pad_pos_enc, dtype=self.tf_int_dtype)
        return (ids_tensor, ppe_tensor)

    def pad_pos_enc(self, pos_enc: List[List[int]]) -> List[List[int]]:
        pad_pos_enc = copy.deepcopy(pos_enc)
        if self.start:
            pad_pos_enc.insert(0, [])
        if self.eos:
            pad_pos_enc.append([])
        if self.pad is not None:
            if self.pad < len(pad_pos_enc):
                raise TokenizationException("Token TPE padding")
            pad_pos_enc.extend([[]] * (self.pad - len(pad_pos_enc)))
        if self.pos_pad is not None:
            for l in pad_pos_enc:
                if self.pos_pad < len(l):
                    raise TokenizationException("TPE padding")
                l.extend([0] * (self.pos_pad - len(l)))
        return pad_pos_enc
