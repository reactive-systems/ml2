"""Simple string data type class"""


from .dtype import DType


class String(DType):
    def __init__(self, string: str) -> None:
        self._string = string

    def to_str(self, **kwargs) -> str:
        return self._string

    def size(self, **kwargs) -> int:
        return len(self.to_str(**kwargs))

    @classmethod
    def from_str(cls, string: str, **kwargs) -> "String":
        return cls(string=string)
