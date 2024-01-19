"""Tokenizer that encodes to decomposed sequence encoding with positional encoding"""

import logging
from typing import Generic, List, Tuple, TypeVar

import tensorflow as tf

from ..dtypes import DType
from ..registry import register_type
from .to_decomp_seq_tokenizer import ToDecompSeqTokenizer
from .to_seq_pos_tokenizer import SeqPosEncoding, ToSeqPosTokenizer
from .tokenizer import TFEncoding

S = TypeVar("S", bound=DType)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecompSeqPosEncoding:
    def __init__(self, seq_pos_encs: List[SeqPosEncoding]) -> None:
        self.seq_pos_encs = seq_pos_encs


@register_type
class ToDecompSeqPosTokenizer(ToDecompSeqTokenizer[S, T], Generic[S, T]):
    def __init__(self, sub_tokenizer: ToSeqPosTokenizer[S], **kwargs):
        super().__init__(sub_tokenizer=sub_tokenizer, **kwargs)

    @property
    def tf_signature(self) -> Tuple[tf.TypeSpec, tf.TypeSpec]:
        sub_seq_spec = tf.TensorSpec(
            shape=(self.num_sub_seqs, self.sub_tokenizer.pad),
            dtype=self.sub_tokenizer.tf_int_dtype,
        )
        sub_pe_spec = tf.TensorSpec(
            shape=(
                self.num_sub_seqs,
                self.sub_tokenizer.pad,
                self.sub_tokenizer.pos_pad,
            ),
            dtype=self.sub_tokenizer.tf_int_dtype,
        )
        return (sub_seq_spec, sub_pe_spec)

    def encode_tf(self, data: T, **kwargs) -> TFEncoding:
        decomp_seq_pos_enc = self.encode(data, **kwargs)
        decomp_ids_tensor = tf.constant(
            [spe.ids for spe in decomp_seq_pos_enc.seq_pos_encs],
            dtype=self.sub_tokenizer.tf_int_dtype,
        )
        decomp_ppe_tensor = tf.constant(
            [spe.pad_pos_enc for spe in decomp_seq_pos_enc.seq_pos_encs],
            dtype=self.sub_tokenizer.tf_int_dtype,
        )
        return (decomp_ids_tensor, decomp_ppe_tensor)
