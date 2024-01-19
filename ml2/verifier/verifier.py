"""Abstract verifier class"""

from abc import abstractmethod
from typing import Generic, TypeVar

from ..configurable import Configurable
from ..dtypes import DType

P = TypeVar("P", bound=DType)
S = TypeVar("S", bound=DType)


class Verifier(Configurable, Generic[P, S]):
    @abstractmethod
    def verify(self, problem: P, solution: S, **kwargs):
        raise NotImplementedError()
