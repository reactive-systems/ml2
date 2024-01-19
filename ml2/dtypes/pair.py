"""Pair data type classes"""

from abc import abstractmethod
from typing import Generic, TypeVar

from .dtype import DType

T1 = TypeVar("T1", bound=DType)
T2 = TypeVar("T2", bound=DType)


class Pair(DType, Generic[T1, T2]):
    def __getitem__(self, key):
        if key == 0:
            return self.fst
        elif key == 1:
            return self.snd
        else:
            raise ValueError(f"Invalid key {key}")

    @property
    @abstractmethod
    def fst(self) -> T1:
        raise NotImplementedError()

    @property
    @abstractmethod
    def snd(self) -> T2:
        raise NotImplementedError()

    def size(self, **kwargs) -> int:
        return self.fst.size(**kwargs) + self.snd.size(**kwargs)

    @abstractmethod
    def swap(self) -> "Pair[T2, T1]":
        raise NotImplementedError()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.fst == other.fst and self.snd == other.snd
        return False

    @classmethod
    @abstractmethod
    def from_components(cls, fst: T1, snd: T2, **kwargs) -> "Pair[T1,T2]":
        raise NotImplementedError()


class GenericPair(Pair[T1, T2], Generic[T1, T2]):
    def __init__(self, fst: T1, snd: T2) -> None:
        self._fst = fst
        self._snd = snd

    @property
    def fst(self) -> T1:
        return self._fst

    @property
    def snd(self) -> T2:
        return self._snd

    def swap(self) -> "GenericPair[T2, T1]":
        return GenericPair(fst=self.snd, snd=self.fst)

    @classmethod
    def from_components(cls, fst: T1, snd: T2, **kwargs) -> "GenericPair[T1, T2]":
        return GenericPair(fst=fst, snd=snd)
