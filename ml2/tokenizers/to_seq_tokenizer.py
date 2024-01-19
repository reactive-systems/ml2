"""Tokenizer that encodes into a sequence encoding"""


from abc import abstractmethod
from typing import Generator, Generic, List, Set, TypeVar

import tensorflow as tf
from tqdm import tqdm

from ..dtypes import DType
from ..registry import register_type
from .tokenizer import (
    EOS_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    Encoding,
    TokenizationException,
    Tokenizer,
)
from .vocabulary import Vocabulary

T = TypeVar("T", bound=DType)


class SeqEncoding(Encoding):
    def __init__(self, tokens: List[str], pad_tokens: List[str], ids: List[int]) -> None:
        self.pad_tokens = pad_tokens
        super().__init__(tokens=tokens, ids=ids)


@register_type
class ToSeqTokenizer(Tokenizer[T], Generic[T]):
    def __init__(
        self,
        pad: int = None,
        eos: bool = False,
        start: bool = False,
        **kwargs,
    ):
        """
        Args:
            start: whether to include start token
            eos: whether to include end of string token
            pad: length to which encoding is padded or None for no padding
            vocabulary: vocabulary object, optional
        """
        self.start = start
        self.eos = eos
        self.pad = pad

        super().__init__(**kwargs)

    @property
    def tf_signature(self) -> tf.TypeSpec:
        return tf.TensorSpec(shape=(self.pad,), dtype=self.tf_int_dtype)

    def encode(self, data: T, **kwargs) -> SeqEncoding:
        tokens = self.encode_tokens(data, **kwargs)
        pad_tokens = self.add_special_tokens(tokens)
        ids = self.vocabulary.tokens_to_ids(pad_tokens)
        if None in ids:
            raise TokenizationException(f"Unknown token {pad_tokens[ids.index(None)]}")
        return SeqEncoding(tokens=tokens, pad_tokens=pad_tokens, ids=ids)

    def encode_tf(self, data: T, **kwargs) -> tf.Tensor:
        encoding = self.encode(data, **kwargs)
        return tf.constant(encoding.ids, dtype=self.tf_int_dtype)

    @abstractmethod
    def encode_tokens(self, data: T, **kwargs) -> List[str]:
        raise NotImplementedError()

    def decode(self, ids: List[int], **kwargs) -> T:
        pad_tokens = self.vocabulary.ids_to_tokens(ids)
        if None in pad_tokens:
            raise Exception(f"Unknown id {ids[pad_tokens.index(None)]}")
        tokens = self.remove_special_tokens(pad_tokens)
        return self.decode_tokens(tokens, **kwargs)

    def decode_tf(self, encoding: tf.Tensor, **kwargs) -> T:
        ids = encoding.numpy().tolist()
        return self.decode(ids=ids, **kwargs)

    @abstractmethod
    def decode_tokens(self, tokens: List[str], **kwargs) -> T:
        raise NotImplementedError()

    def sort_tokens(self, tokens: List[str]) -> None:
        tokens.sort()

    def vocabulary_filename(self) -> str:
        filename = super().vocabulary_filename()
        if self.start:
            filename += "-s"
        if self.eos:
            filename += "-e"
        if self.pad is not None:
            filename += "-p"
        return filename

    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        pad_tokens = tokens.copy()
        if self.start:
            pad_tokens.insert(0, START_TOKEN)
        if self.eos:
            pad_tokens.append(EOS_TOKEN)
        if self.pad is not None:
            if self.pad < len(pad_tokens):
                raise TokenizationException("Exceeding padding length")
            pad_tokens.extend([PAD_TOKEN] * (self.pad - len(pad_tokens)))
        return pad_tokens

    def remove_special_tokens(self, pad_tokens: List[str]) -> List[str]:
        tokens = pad_tokens.copy()
        for _ in range(len(tokens)):
            if tokens[-1] == PAD_TOKEN:
                tokens.pop()
            else:
                break
        if tokens and tokens[-1] == EOS_TOKEN:
            tokens.pop()
        if tokens and tokens[0] == START_TOKEN:
            tokens.pop(0)
        return tokens

    def build_vocabulary(
        self,
        generator: Generator[T, None, None],
        add_start: bool = False,
        add_eos: bool = False,
        add_pad: bool = False,
        **kwargs,
    ) -> None:
        token_set: Set[str] = set()
        pbar = tqdm(desc="Building vocabulary", unit="sample")
        for data in generator:
            pbar.update()
            tokens = self.encode_tokens(data, **kwargs)
            token_set = token_set.union(tokens)
        pbar.close()
        token_list = list(token_set)
        self.sort_tokens(token_list)
        if self.start or add_start:
            token_list.append(START_TOKEN)
        if self.eos or add_eos:
            token_list.append(EOS_TOKEN)
        if self.pad is not None or add_pad:
            # putting pad token at the beginning ensures that pad token id corresponds to zero
            token_list = [PAD_TOKEN] + token_list
        self._vocabulary = Vocabulary.from_iterable(
            token_list, name=self.name + "/vocab", project=self.project
        )
