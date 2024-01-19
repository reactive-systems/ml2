"""Tokenizer that encodes to decomposed sequence encoding"""

import logging
from typing import Generic, List, Tuple, TypeVar

import tensorflow as tf

from ..dtypes import DType
from .to_seq_tokenizer import SeqEncoding, ToSeqTokenizer
from .tokenizer import TFEncoding, Tokenizer
from .vocabulary import Vocabulary

S = TypeVar("S", bound=DType)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecompSeqEncoding:
    def __init__(self, seq_encs: List[SeqEncoding]) -> None:
        self.seq_encs = seq_encs


class ToDecompSeqTokenizer(Tokenizer[T], Generic[S, T]):
    def __init__(self, sub_tokenizer: ToSeqTokenizer[S], num_sub_seqs: int = None, **kwargs):
        self.num_sub_seqs = num_sub_seqs
        self.sub_tokenizer = sub_tokenizer
        super().__init__(**kwargs)

    @property
    def tf_signature(self) -> Tuple[tf.TypeSpec, tf.TypeSpec]:
        return tf.TensorSpec(
            shape=(self.num_sub_seqs, self.sub_tokenizer.pad),
            dtype=self.sub_tokenizer.tf_int_dtype,
        )

    @property
    def vocabulary(self) -> Vocabulary:
        return self.sub_tokenizer.vocabulary

    def encode_tf(self, data: T, **kwargs) -> TFEncoding:
        decomp_seq_enc = self.encode(data, **kwargs)
        return tf.constant(
            [se.ids for se in decomp_seq_enc.seq_encs],
            dtype=self.sub_tokenizer.tf_int_dtype,
        )

    def save(
        self,
        add_to_wandb: bool = False,
        overwrite_bucket: bool = False,
        overwrite_local: bool = False,
        recurse: bool = False,
        upload: bool = False,
    ) -> None:
        super().save(
            add_to_wandb=add_to_wandb,
            overwrite_bucket=overwrite_bucket,
            overwrite_local=overwrite_local,
            upload=upload,
        )

        if recurse:
            self.sub_tokenizer.save(
                add_to_wandb=add_to_wandb,
                overwrite_bucket=overwrite_bucket,
                overwrite_local=overwrite_local,
                recurse=recurse,
                upload=upload,
            )
