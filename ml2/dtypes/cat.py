"""Abstract categorical data type class"""

from abc import abstractmethod

from .dtype import DType


class Cat(DType):
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.token() == other.token()
        return False

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.token()}>"

    def size(self, **kwargs) -> int:
        return 1

    @abstractmethod
    def token(self, **kwargs) -> str:
        raise NotImplementedError()

    @property
    def value(self) -> str:
        return self.token()

    @classmethod
    @abstractmethod
    def from_token(cls, token: str, **kwargs) -> "Cat":
        raise NotImplementedError()
