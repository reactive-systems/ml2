"""Abstract encoder class"""

from typing import Generator, List

START_TOKEN = "<s>"
EOS_TOKEN = "<e>"
PAD_TOKEN = "<p>"


class Encoder:
    def __init__(self):
        self.tokens = None
        self.ids = None
        self.error = None

    def tokenize(self) -> bool:
        raise NotImplementedError

    def detokenize(self) -> bool:
        raise NotImplementedError

    def encode(self, input) -> bool:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> bool:
        raise NotImplementedError

    def load_vocabulary(self, dir: str) -> bool:
        raise NotImplementedError

    def build_vocabulary(self, generator: Generator) -> None:
        raise NotImplementedError

    def vocabulary_to_file(self, dir: str) -> None:
        raise NotImplementedError

    def vocabulary_filename(self) -> str:
        raise NotImplementedError
