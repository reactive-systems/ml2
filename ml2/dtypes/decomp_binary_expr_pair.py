"""Pair of decomposed binary expressions"""

from abc import abstractmethod
from typing import Generic, List, TypeVar

from .binary_ast import BinaryAST
from .binary_expr import BinaryExpr
from .decomp_binary_expr import DecompBinaryExpr
from .pair import GenericPair

T = TypeVar("T", bound=BinaryExpr)


class DecompBinaryExprPair(
    BinaryExpr, GenericPair[DecompBinaryExpr[T], DecompBinaryExpr[T]], Generic[T]
):
    def __init__(self, fst: DecompBinaryExpr[T], snd: DecompBinaryExpr[T]):
        GenericPair.__init__(self, fst=fst, snd=snd)

    @property
    def ast(self) -> BinaryAST:
        return self.comp_ast_pair(self[0].ast, self[1].ast)

    def to_tokens(self, notation: str = None, **kwargs) -> List[str]:
        if notation is None and self[0]._notation is not None:
            notation = self[0]._notation
        if notation is None and self[1]._notation is not None:
            notation = self[1]._notation
        if notation is None:
            notation = "infix"
        return self.comp_token_list_pair(
            self[0].to_tokens(notation=notation),
            self[1].to_tokens(notation=notation),
            notation=notation,
        )

    def to_str(self, notation: str = None) -> str:
        if notation is None and len(self[0].sub_exprs) > 0:
            notation = self[0].sub_exprs[0]._notation
        if notation is None and len(self[1].sub_exprs) > 0:
            notation = self[1].sub_exprs[0]._notation
        if len(self[0].sub_exprs) > 0 and len(self[1].sub_exprs) > 0:
            return self.comp_str_pair(
                self[0].to_str(notation=notation),
                self[1].to_str(notation=notation),
                notation=notation,
            )
        elif len(self[0].sub_exprs) > 0:
            return self[0].to_str(notation=notation)
        elif len(self[1].sub_exprs) > 0:
            return self[1].to_str(notation=notation)
        else:
            return ""

    @classmethod
    def from_ast(cls, ast: BinaryAST, **kwargs) -> "DecompBinaryExprPair[T]":
        raise NotImplementedError()

    @classmethod
    def from_str(cls, s: str, notation: str = "infix", **kwargs) -> "DecompBinaryExprPair[T]":
        raise NotImplementedError()

    @classmethod
    def from_tokens(
        cls, tokens: List[str], notation: str = "infix", **kwargs
    ) -> "DecompBinaryExprPair[T]":
        raise NotImplementedError()

    @classmethod
    def from_asts(
        cls, ast_list1: List[BinaryAST], ast_list2: List[BinaryAST], **kwargs
    ) -> "DecompBinaryExprPair[T]":
        raise NotImplementedError()

    @classmethod
    def from_strs(
        cls, str_list1: List[str], str_list2: List[str], notation: str = "infix", **kwargs
    ) -> "DecompBinaryExprPair[T]":
        raise NotImplementedError()

    @classmethod
    def from_token_lists(
        cls,
        token_lists1: List[List[str]],
        token_lists2: List[List[str]],
        notation: str = "infix",
        **kwargs
    ) -> "DecompBinaryExprPair[T]":
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def comp_ast_pair(ast1: BinaryAST, ast2: BinaryAST) -> BinaryAST:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def comp_str_pair(str1: str, str2: str, notation: str = None) -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def comp_token_list_pair(
        token_list1: List[str], token_list2: List[str], notation: str = None
    ) -> List[str]:
        raise NotImplementedError()
