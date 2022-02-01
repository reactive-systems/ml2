"""Propositional formula"""

from typing import List

from ..data import BinaryAST, ExprNotation
from .prop_lexer import lex_prop
from .prop_parser import parse_infix_prop, parse_prefix_prop


class PropFormula:
    def __init__(
        self,
        ast: BinaryAST = None,
        formula: str = None,
        notation: str = None,
        tokens: List[str] = None,
    ):
        self._ast = ast
        self._str = formula
        self._notation = notation
        self._tokens = tokens

    @property
    def ast(self) -> BinaryAST:
        if not self._ast:
            if self._notation == "infix":
                self._ast = parse_infix_prop(self._str)
            elif self._notation == "prefix":
                self._ast = parse_prefix_prop(self._str)
            else:
                raise ValueError(f"Initialized with invalid notation {self._notation}")
        return self._ast

    def to_str(self, notation: str = None) -> str:
        if not notation or notation == self._notation:
            return self._str
        elif notation == "infix":
            return "".join(self.ast.to_list(ExprNotation.INFIX))
        elif notation == "prefix":
            return " ".join(self.ast.to_list(ExprNotation.PREFIX))
        else:
            raise ValueError(f"Invalid notation {notation}")

    def tokens(self, notation: str = None) -> List[str]:
        tokens = []
        if not notation:
            notation = self._notation

        if notation == self._notation and self._tokens:
            return self._tokens
        elif notation == self._notation:
            tokens = lex_prop(self.to_str())
        elif notation == "infix":
            tokens = self.ast.to_list(notation=ExprNotation.INFIX)
        elif notation == "prefix":
            tokens = self.ast.to_list(notation=ExprNotation.PREFIX)
        else:
            raise ValueError(f"Invalid notation {notation}")

        if notation == self._notation:
            self._tokens = tokens

        return tokens

    def size(self) -> int:
        return self._ast.size()

    @classmethod
    def from_ast(cls, ast: BinaryAST):
        return cls(ast=ast)

    @classmethod
    def from_str(cls, formula: str, notation: str = "infix"):
        return cls(formula=formula, notation=notation)
