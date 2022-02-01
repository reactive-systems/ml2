"""LTL Formula"""

from typing import List

from ..data import BinaryAST, ExprNotation
from .ltl_lexer import lex_ltl
from .ltl_parser import parse_infix_ltl, parse_prefix_ltl


class LTLFormula:
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
                self._ast = parse_infix_ltl(self._str)
            elif self._notation == "prefix":
                self._ast = parse_prefix_ltl(self._str)
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

    @property
    def tokens(self) -> List[str]:
        if not self._tokens:
            self._tokens = lex_ltl(self._str)
        return self._tokens

    def size(self) -> int:
        return self._ast.size()

    @classmethod
    def from_ast(cls, ast: BinaryAST):
        return cls(ast=ast)

    @classmethod
    def from_str(cls, formula: str, notation: str = "infix"):
        return cls(formula=formula, notation=notation)
