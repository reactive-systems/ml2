"""Abstract equivalence verifier class"""

from abc import abstractmethod
from typing import Generic, TypeVar

from ..configurable import Configurable
from ..dtypes import DType
from .equiv_status import EquivStatus

T = TypeVar("T", bound=DType)


class EquivVerifier(Configurable, Generic[T]):
    @abstractmethod
    def verify_equiv(self, x: T, y: T, **kwargs) -> EquivStatus:
        raise NotImplementedError()
