"""Abstract equivalence status class"""

from abc import abstractmethod
from typing import Optional

from ..dtypes import DType


class EquivStatus(DType):
    @property
    @abstractmethod
    def equiv(self) -> Optional[bool]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError()
