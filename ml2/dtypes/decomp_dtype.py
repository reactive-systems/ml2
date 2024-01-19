"""Decomposed data type classes"""

from abc import abstractmethod
from typing import Generic, List, TypeVar

from .dtype import DType

T = TypeVar("T", bound=DType)


class DecompDType(DType, Generic[T]):
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> T:
        raise NotImplementedError()

    @property
    @abstractmethod
    def len(self) -> int:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_components(cls, components: List[T], **kwargs) -> "DecompDType[T]":
        raise NotImplementedError()


class GenericDecompDType(list, DecompDType[T], Generic[T]):
    def __init__(
        self,
        components: List[T] = None,
    ) -> None:
        super().__init__(components if components is not None else [])

    def size(self, **kwargs) -> int:
        return sum([c.size(**kwargs) for c in self])

    @classmethod
    def from_components(cls, components: List[T], **kwargs) -> "GenericDecompDType[T]":
        return cls(components=components)
