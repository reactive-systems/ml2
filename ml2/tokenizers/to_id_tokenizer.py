"""Tokenizer that encodes into an index"""


from abc import abstractmethod
from typing import Generator, Generic, List, Set, TypeVar

import tensorflow as tf
from tqdm import tqdm

from ..dtypes import DType
from ..registry import register_type
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary

T = TypeVar("T", bound=DType)


@register_type
class ToIdTokenizer(Tokenizer[T], Generic[T]):
    @property
    def tf_signature(self) -> tf.TypeSpec:
        return tf.TensorSpec(shape=(), dtype=self.tf_int_dtype)

    def encode(self, data: T, **kwargs) -> int:
        token = self.encode_token(data, **kwargs)
        return self.vocabulary.token_to_id[token]

    def encode_tf(self, data: T, **kwargs) -> tf.Tensor:
        idx = self.encode(data, **kwargs)
        return tf.constant(idx, dtype=self.tf_int_dtype)

    @abstractmethod
    def encode_token(self, data: T, **kwargs) -> str:
        raise NotImplementedError()

    def decode(self, idx: int, **kwargs) -> T:
        token = self.vocabulary.id_to_token(idx)
        return self.decode_token(token, **kwargs)

    def decode_tf(self, encoding: tf.Tensor, **kwargs) -> T:
        idx = encoding.numpy().tolist()
        return self.decode(idx=idx, **kwargs)

    @abstractmethod
    def decode_token(self, token: str, **kwargs) -> T:
        raise NotImplementedError()

    def sort_tokens(self, tokens: List[str]) -> None:
        tokens.sort()

    def build_vocabulary(self, generator: Generator[T, None, None], **kwargs) -> None:
        token_set: Set[str] = set()
        pbar = tqdm(desc="Building vocabulary", unit="sample")
        for data in generator:
            pbar.update()
            token = self.encode_token(data, **kwargs)
            token_set.add(token)
        pbar.close()
        token_list = list(token_set)
        self.sort_tokens(token_list)
        self._vocabulary = Vocabulary.from_iterable(
            token_list, name=self.name + "/vocab", project=self.project
        )
