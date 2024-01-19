"""Abstract dataset writer class"""

from abc import abstractmethod
from typing import Generic, TypeVar

from ..artifact import Artifact
from ..dtypes import DType

T = TypeVar("T", bound=DType)


class DatasetWriter(Artifact, Generic[T]):
    @abstractmethod
    def add_sample(self, sample: T, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def size(self, **kwargs) -> int:
        raise NotImplementedError()
