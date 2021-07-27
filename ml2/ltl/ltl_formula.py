"""LTL Formula"""

from .ltl_lexer import lex_ltl
from .ltl_parser import parse_infix_ltl, parse_prefix_ltl


class LTLFormula():

    def __init__(self,
                 ast=None,
                 formula: str = None,
                 notation: str = None,
                 tokens: list = None):
        self._ast = ast
        self._str = formula
        self._notation = notation
        self._tokens = tokens

    @property
    def ast(self):
        if not self._ast:
            if self._notation == 'infix':
                self._ast = parse_infix_ltl(self._str)
            elif self._notation == 'prefix':
                self._ast = parse_prefix_ltl(self._str)
            else:
                raise ValueError(f'Invalid notation {self._notation}')
        return self._ast

    @property
    def tokens(self):
        if not self._tokens:
            self._tokens = lex_ltl(self._str)
        return self._tokens

    def size(self):
        return self._ast.size()

    @classmethod
    def from_ast(cls, ast):
        return cls(ast=ast)

    @classmethod
    def from_str(cls, formula: str, notation: str = None):
        return cls(formula=formula, notation=notation)
