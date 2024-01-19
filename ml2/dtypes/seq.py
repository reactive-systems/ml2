"""Abstract sequence data type class"""

from abc import abstractmethod
from typing import List

from .dtype import DType


class Seq(DType):
    @abstractmethod
    def to_tokens(self, **kwargs) -> List[str]:
        raise NotImplementedError()

    def size(self, **kwargs) -> int:
        return len(self.to_tokens(**kwargs))

    @classmethod
    @abstractmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "Seq":
        raise NotImplementedError()
