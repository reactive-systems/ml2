"""Supervised data type classes"""

from abc import abstractmethod
from typing import Generic, TypeVar

from .dtype import DType
from .pair import GenericPair, Pair

# input type variable
I = TypeVar("I", bound=DType)
# target type variable
T = TypeVar("T", bound=DType)


class Supervised(Pair[I, T], Generic[I, T]):
    @property
    def fst(self) -> I:
        return self.input

    @property
    def snd(self) -> T:
        return self.target

    @property
    @abstractmethod
    def input(self) -> I:
        raise NotImplementedError()

    @property
    @abstractmethod
    def target(self) -> T:
        raise NotImplementedError()

    def __getitem__(self, key):
        if key == 0:
            return self.input
        elif key == 1:
            return self.target
        else:
            raise IndexError("Index out of range or no integer.")


# inheritance order such that properties first and second are inherited from GenericPair and not Supervised
class GenericSupervised(GenericPair[I, T], Supervised[I, T], Generic[I, T]):
    def __init__(self, input: I, target: T) -> None:
        super().__init__(fst=input, snd=target)

    @property
    def input(self) -> I:
        return self.fst

    @property
    def target(self) -> T:
        return self.snd
