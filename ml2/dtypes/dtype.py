"""Abstract data type class"""

from abc import abstractmethod


class DType(object):
    @abstractmethod
    def size(self, **kwargs) -> int:
        raise NotImplementedError()
