"""propositional logic parser"""

import sly

from ..data.ast import BinaryAST
from .prop_lexer import PropPrefixLexer, PropInfixLexer


class PropPrefixParser(sly.Parser):

    tokens = PropPrefixLexer.tokens
    precedence = (
        ("right", EQUIV, IMPL),
        ("left", XOR),
        ("left", OR),
        ("left", AND),
        ("right", NOT),
    )

    @_("EQUIV expr expr", "IMPL expr expr", "XOR expr expr", "OR expr expr", "AND expr expr")
    def expr(self, p):
        return BinaryAST(p[0], p.expr0, p.expr1)

    @_("NOT expr")
    def expr(self, p):
        return BinaryAST(p[0], p.expr)

    @_("CONST")
    def expr(self, p):
        return BinaryAST(p.CONST)

    @_("V")
    def expr(self, p):
        return BinaryAST(p.V)

    def error(self, p):
        pass


class PropInfixParser(sly.Parser):

    tokens = PropInfixLexer.tokens
    precedence = (
        ("right", EQUIV, IMPL),
        ("left", XOR),
        ("left", OR),
        ("left", AND),
        ("right", NOT),
    )

    @_("expr EQUIV expr", "expr IMPL expr", "expr XOR expr", "expr OR expr", "expr AND expr")
    def expr(self, p):
        return BinaryAST(p[1], p.expr0, p.expr1)

    @_("NOT expr")
    def expr(self, p):
        return BinaryAST(p[0], p.expr)

    @_("LPAR expr RPAR")
    def expr(self, p):
        return p.expr

    @_("CONST")
    def expr(self, p):
        return BinaryAST(p.CONST)

    @_("V")
    def expr(self, p):
        return BinaryAST(p.V)

    def error(self, p):
        pass


INFIX_LEXER = None
INFIX_PARSER = None
PREFIX_LEXER = None
PREFIX_PARSER = None


def parse_infix_prop(formula: str):
    global INFIX_LEXER
    if INFIX_LEXER is None:
        INFIX_LEXER = PropInfixLexer()
    global INFIX_PARSER
    if INFIX_PARSER is None:
        INFIX_PARSER = PropInfixParser()
    return INFIX_PARSER.parse(INFIX_LEXER.tokenize(formula))


def parse_prefix_prop(formula: str):
    global PREFIX_LEXER
    if PREFIX_LEXER is None:
        PREFIX_LEXER = PropPrefixLexer()
    global PREFIX_PARSER
    if PREFIX_PARSER is None:
        PREFIX_PARSER = PropPrefixParser()
    return PREFIX_PARSER.parse(PREFIX_LEXER.tokenize(formula))


def parse_prop(formula: str):
    """
    Parses LTL formula
    Args:
        formula: string, in infix or prefix notation
    Returns:
        abstract syntax tree or None if formula can not be parsed
    """
    ast = parse_infix_prop(formula)
    if ast:
        return ast
    return parse_prefix_prop(formula)
