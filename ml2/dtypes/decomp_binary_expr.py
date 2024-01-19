"""Decomposed binary expression"""

from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Type, TypeVar

from .binary_ast import BinaryAST
from .binary_expr import BinaryExpr
from .decomp_dtype import GenericDecompDType

T = TypeVar("T", bound=BinaryExpr)


class DecompBinaryExpr(GenericDecompDType[T], BinaryExpr, Generic[T]):
    BINARY_EXPR_TYPE: Type[T] = None

    def __init__(
        self,
        sub_exprs: List[T] = None,
    ) -> None:
        GenericDecompDType.__init__(self, components=sub_exprs)

    @property
    def sub_exprs(self) -> List[T]:
        return self

    @property
    def _notation(self) -> Optional[str]:
        if len(self) > 0:
            return self[0]._notation
        return None

    @property
    def ast(self) -> BinaryAST:
        return self.comp_asts([se.ast for se in self])

    def to_tokens(self, notation: str = None, **kwargs) -> List[str]:
        if notation is None:
            notation = self._notation if self._notation is not None else "infix"
        return self.comp_token_lists(
            token_lists=[se.to_tokens(notation=notation, **kwargs) for se in self],
            notation=notation,
        )

    def to_str(self, notation: Optional[str] = None) -> str:
        if notation is None and len(self) > 0:
            notation = self[0]._notation
        return self.comp_strs(strs=self.to_strs(notation=notation), notation=notation)

    def to_strs(self, notation: Optional[str] = None) -> List[str]:
        if notation is None and len(self) > 0:
            notation = self[0]._notation
        return [se.to_str(notation) for se in self]

    def rename(self, rename: Dict[str, str]):
        for expr in self.sub_exprs:
            expr.rename(rename)

    @classmethod
    def from_ast(cls, ast: BinaryAST, **kwargs) -> "DecompBinaryExpr[T]":
        return cls.from_asts(asts=[ast], **kwargs)

    @classmethod
    def from_str(cls, s: str, notation: str = "infix", **kwargs) -> "DecompBinaryExpr[T]":
        return cls.from_strs(strs=[s], notation=notation, **kwargs)

    @classmethod
    def from_tokens(
        cls, tokens: List[str], notation: str = "infix", **kwargs
    ) -> "DecompBinaryExpr[T]":
        return cls.from_token_lists(token_lists=[tokens], notation=notation, **kwargs)

    @classmethod
    def from_asts(cls, asts: List[BinaryAST], **kwargs) -> "DecompBinaryExpr[T]":
        return cls(
            sub_exprs=[cls.BINARY_EXPR_TYPE.from_ast(ast, **kwargs) for ast in asts], **kwargs
        )

    @classmethod
    def from_strs(
        cls, strs: List[str], notation: str = "infix", **kwargs
    ) -> "DecompBinaryExpr[T]":
        return cls(
            sub_exprs=[
                cls.BINARY_EXPR_TYPE.from_str(s, notation=notation, **kwargs) for s in strs
            ],
            **kwargs
        )

    @classmethod
    def from_token_lists(
        cls, token_lists: List[List[str]], notation: str = "infix", **kwargs
    ) -> "DecompBinaryExpr[T]":
        return cls(
            sub_exprs=[
                cls.BINARY_EXPR_TYPE.from_tokens(tokens=tokens, notation=notation, **kwargs)
                for tokens in token_lists
            ],
            **kwargs
        )

    @staticmethod
    @abstractmethod
    def comp_asts(asts: List[BinaryAST]) -> BinaryAST:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def comp_strs(strs: List[str], notation: Optional[str] = None) -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def comp_token_lists(
        token_lists: List[List[str]], notation: Optional[str] = None
    ) -> List[str]:
        raise NotImplementedError()
