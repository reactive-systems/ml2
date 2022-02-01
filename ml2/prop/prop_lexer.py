"""Propositional logic lexer"""

import logging
import sly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropPrefixLexer(sly.Lexer):

    ops = {NOT, AND, OR, XOR, IMPL, EQUIV}
    tokens = {V, CONST}.union(ops)

    ignore = " \t"

    CONST = r"true|false|1|0"
    V = r"[a-zA-Z_][a-zA-Z0-9_]*"

    NOT = r"!"
    AND = r"&(&)?"
    OR = r"\|(\|)?"
    XOR = r"\^"
    IMPL = r"->"
    EQUIV = r"<->"

    def error(self, t):
        # TODO figure out how to return None instead of skipping illegal characters
        logger.debug(f"Illegal character {t.value[0]}")
        self.index += 1


class PropInfixLexer(sly.Lexer):

    ops = {NOT, AND, OR, XOR, IMPL, EQUIV}
    tokens = {V, CONST, LPAR, RPAR}.union(ops)

    ignore = " \t"

    CONST = r"true|false|1|0"
    V = r"[a-zA-Z_][a-zA-Z0-9_]*"
    LPAR = r"\("
    RPAR = r"\)"

    NOT = r"!"
    AND = r"&(&)?"
    OR = r"\|(\|)?"
    XOR = r"\^"
    IMPL = r"->"
    EQUIV = r"<->"

    def error(self, t):
        logger.debug(f"Illegal character {t.value[0]}")
        self.index += 1


PROP_INFIX_LEXER = None


def lex_prop(formula: str) -> list:
    global PROP_INFIX_LEXER
    if PROP_INFIX_LEXER is None:
        PROP_INFIX_LEXER = PropInfixLexer()
    return [token.value for token in PROP_INFIX_LEXER.tokenize(formula)]
