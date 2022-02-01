"""Sequence encoder"""

import tensorflow as tf
from tqdm import tqdm
from typing import Generator, List

from .encoder import Encoder, START_TOKEN, EOS_TOKEN, PAD_TOKEN
from ..vocabulary import Vocabulary


class SeqEncoder(Encoder):
    def __init__(
        self,
        start: bool,
        eos: bool,
        pad: int,
        encode_start: bool = True,
        vocabulary: Vocabulary = None,
        tf_dtype: tf.DType = tf.int32,
    ):
        """
        Args:
            start: whether to include start token
            eos: whether to include end of string token
            pad: length to which encoding is padded or None for no padding
            encode_start: whether to include start token in encodings
            vocabulary: vocabulary object, optional
            tf_dtype: datatype of tensor encodings
        """
        self.start = start
        self.eos = eos
        self.pad = pad
        self.encode_start = encode_start
        self.vocabulary = vocabulary
        self.tf_dtype = tf_dtype
        #
        self.sequence = None
        self.tokens = None
        self.padded_tokens = None
        self.error = None
        super().__init__()

    def lex(self) -> bool:
        raise NotImplementedError

    @property
    def tensor_spec(self):
        return tf.TensorSpec(shape=(self.pad,), dtype=self.tf_dtype)

    @property
    def tensor(self):
        return tf.constant(self.ids, dtype=self.tf_dtype)

    def tokenize(self) -> bool:
        return self.lex()

    def detokenize(self) -> bool:
        self.sequence = " ".join(self.tokens)
        return True

    def encode(self, sequence) -> bool:
        self.sequence = sequence
        success = self.tokenize()
        success = success and self.add_special_tokens()
        if success and self.vocabulary is not None:
            self.ids = self.vocabulary.tokens_to_ids(self.padded_tokens)
            if None in self.ids:
                self.error = f"Unknown token {self.padded_tokens[self.ids.index(None)]}"
                success = False
        return success

    def decode(self, ids: List[int]) -> bool:
        self.padded_tokens = self.vocabulary.ids_to_tokens(ids)
        if None in self.padded_tokens:
            self.error = f"Unknown id {ids[self.padded_tokens.index(None)]}"
            success = False
        else:
            success = True
        success = success and self.remove_special_tokens()
        success = success and self.detokenize()
        return success

    def load_vocabulary(self, dir: str) -> bool:
        filename = self.vocabulary_filename()
        self.vocabulary = Vocabulary.from_file(dir, filename)
        return self.vocabulary is not None

    def sort_tokens(self, tokens: list) -> None:
        tokens.sort()

    def build_vocabulary(self, generator: Generator) -> None:
        token_set = set()
        pbar = tqdm(desc="Building vocabulary", unit="sample")
        for sequence in generator:
            pbar.update()
            self.encode(sequence)
            token_set = token_set.union(self.tokens)
        pbar.close()
        token_list = list(token_set)
        self.sort_tokens(token_list)
        if self.start:
            token_list.append(START_TOKEN)
        if self.eos:
            token_list.append(EOS_TOKEN)
        if self.pad:
            # putting pad token at the beginning ensures that pad token id corresponds to zero
            token_list = [PAD_TOKEN] + token_list
        self.vocabulary = Vocabulary.from_iterable(token_list)

    def vocabulary_to_file(self, dir: str) -> None:
        self.vocabulary.to_file(dir, self.vocabulary_filename())

    def vocabulary_filename(self) -> str:
        filename = ""
        if self.start:
            filename += "-s"
        if self.eos:
            filename += "-e"
        if self.pad:
            filename += "-p"
        return filename

    def add_special_tokens(self) -> bool:
        self.padded_tokens = self.tokens.copy()
        if self.start and self.encode_start:
            self.padded_tokens.insert(0, START_TOKEN)
        if self.eos:
            self.padded_tokens.append(EOS_TOKEN)
        if self.pad:
            if self.pad < len(self.padded_tokens):
                self.error = "Token padding"
                return False
            self.padded_tokens.extend([PAD_TOKEN] * (self.pad - len(self.padded_tokens)))
        return True

    def remove_special_tokens(self) -> bool:
        self.tokens = self.padded_tokens.copy()
        for _ in range(len(self.tokens)):
            if self.tokens[-1] == PAD_TOKEN:
                self.tokens.pop()
            else:
                break
        if self.tokens and self.tokens[-1] == EOS_TOKEN:
            self.tokens.pop()
        if self.tokens and self.tokens[0] == START_TOKEN:
            self.tokens.pop(0)
        return True
