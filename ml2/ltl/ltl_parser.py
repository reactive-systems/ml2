"""LTL parser"""

import sly

from ..data.ast import BinaryAST
from .ltl_lexer import LTLPrefixLexer, LTLInfixLexer


class LTLPrefixParser(sly.Parser):

    tokens = LTLPrefixLexer.tokens
    precedence = (
        ("right", EQUIV, IMPL),
        ("left", XOR),
        ("left", OR),
        ("left", AND),
        ("right", UNTIL, WUNTIL, RELEASE),
        ("right", EVEN, GLOB),
        ("right", NEXT),
        ("right", NOT),
    )

    @_(
        "EQUIV expr expr",
        "IMPL expr expr",
        "XOR expr expr",
        "OR expr expr",
        "AND expr expr",
        "UNTIL expr expr",
        "WUNTIL expr expr",
        "RELEASE expr expr",
    )
    def expr(self, p):
        return BinaryAST(p[0], p.expr0, p.expr1)

    @_("EVEN expr", "GLOB expr", "NEXT expr", "NOT expr")
    def expr(self, p):
        return BinaryAST(p[0], p.expr)

    @_("CONST")
    def expr(self, p):
        return BinaryAST(p.CONST)

    @_("AP")
    def expr(self, p):
        return BinaryAST(p.AP)

    def error(self, p):
        pass


class LTLInfixParser(sly.Parser):

    tokens = LTLInfixLexer.tokens
    precedence = (
        ("right", EQUIV, IMPL),
        ("left", XOR),
        ("left", OR),
        ("left", AND),
        ("right", UNTIL, WUNTIL, RELEASE),
        ("right", EVEN, GLOB),
        ("right", NEXT),
        ("right", NOT),
    )

    @_(
        "expr EQUIV expr",
        "expr IMPL expr",
        "expr XOR expr",
        "expr OR expr",
        "expr AND expr",
        "expr UNTIL expr",
        "expr WUNTIL expr",
        "expr RELEASE expr",
    )
    def expr(self, p):
        return BinaryAST(p[1], p.expr0, p.expr1)

    @_("EVEN expr", "GLOB expr", "NEXT expr", "NOT expr")
    def expr(self, p):
        return BinaryAST(p[0], p.expr)

    @_("LPAR expr RPAR")
    def expr(self, p):
        return p.expr

    @_("CONST")
    def expr(self, p):
        return BinaryAST(p.CONST)

    @_("AP")
    def expr(self, p):
        return BinaryAST(p.AP)

    def error(self, p):
        pass


INFIX_LEXER = None
INFIX_PARSER = None
PREFIX_LEXER = None
PREFIX_PARSER = None


def parse_infix_ltl(formula: str):
    global INFIX_LEXER
    if INFIX_LEXER is None:
        INFIX_LEXER = LTLInfixLexer()
    global INFIX_PARSER
    if INFIX_PARSER is None:
        INFIX_PARSER = LTLInfixParser()
    return INFIX_PARSER.parse(INFIX_LEXER.tokenize(formula))


def parse_prefix_ltl(formula: str):
    global PREFIX_LEXER
    if PREFIX_LEXER is None:
        PREFIX_LEXER = LTLPrefixLexer()
    global PREFIX_PARSER
    if PREFIX_PARSER is None:
        PREFIX_PARSER = LTLPrefixParser()
    return PREFIX_PARSER.parse(PREFIX_LEXER.tokenize(formula))


def parse_ltl(formula: str):
    """
    Parses LTL formula
    Args:
        formula: string, in infix or prefix notation
    Returns:
        abstract syntax tree or None if formula can not be parsed
    """
    ast = parse_infix_ltl(formula)
    if ast:
        return ast
    return parse_prefix_ltl(formula)
