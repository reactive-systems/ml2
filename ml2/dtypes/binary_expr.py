"""Abstract binary expression class"""

from abc import abstractmethod
from typing import Dict, List, Optional, Type, TypeVar

from .binary_ast import BinaryAST
from .seq import Seq

T = TypeVar("T", bound="BinaryExpr")


class BinaryExpr(Seq):
    def __init__(
        self,
        ast: BinaryAST = None,
        formula: str = None,
        notation: str = None,
        tokens: List[str] = None,
    ) -> None:
        assert not (ast is None and formula is None and tokens is None)
        self._notation = notation
        self._ast = ast
        self._str = formula
        self._tokens = tokens
        super().__init__()

    @property
    def ast(self) -> BinaryAST:
        if self._ast is None:
            if self._notation is not None:
                self._ast = self.parse(self.to_str(), notation=self._notation)
            else:
                self._ast = self.parse(self.to_str())
        return self._ast

    def size(self, notation: str = None, **kwargs) -> int:
        return self.ast.size(notation=notation, **kwargs)

    def to_tokens(self, notation: str = None, **kwargs) -> List[str]:
        tokens = []
        if notation is None:
            notation = self._notation if self._notation else "infix"
        if notation == self._notation and self._tokens:
            return self._tokens
        elif notation == self._notation:
            tokens = self.lex(self.to_str())
        else:
            tokens = self.ast.to_tokens(notation=notation)

        if notation == self._notation:
            self._tokens = tokens

        return tokens

    def to_str(self, notation: Optional[str] = None, **kwargs) -> str:
        if not notation:
            notation = (
                self._notation if self._notation is not None or self._str is not None else "infix"
            )
        if notation == self._notation and self._str is not None:
            return self._str
        elif notation == self._notation and self._tokens is not None:
            return " ".join(self._tokens)
        else:
            return " ".join(self.ast.to_tokens(notation=notation))

    @classmethod
    def from_ast(cls: Type[T], ast: BinaryAST, **kwargs) -> T:
        return cls(ast=ast)

    @classmethod
    def from_str(cls: Type[T], formula: str, notation: str = "infix", **kwargs) -> T:
        return cls(formula=formula, notation=notation)

    @classmethod
    def from_tokens(cls: Type[T], tokens: List[str], notation: str = "infix", **kwargs) -> T:
        return cls(tokens=tokens, notation=notation)

    @staticmethod
    @abstractmethod
    def lex(expr: str) -> List[str]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def parse(expr: str, notation: str = "infix") -> BinaryAST:
        raise NotImplementedError()

    def rename(self, rename: Dict[str, str]):
        if hasattr(self, "_ast"):
            self.ast
            self._ast.rename(rename=rename)
            self._str = None
