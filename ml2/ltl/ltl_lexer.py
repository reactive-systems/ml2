"""LTL lexer"""

import logging
import sly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLPrefixLexer(sly.Lexer):

    bool_ops = {NOT, AND, OR, XOR, IMPL, EQUIV}
    temp_ops = {NEXT, UNTIL, WUNTIL, RELEASE, EVEN, GLOB}
    tokens = {AP, CONST}.union(bool_ops, temp_ops)

    ignore = " \t"

    CONST = r"true|false|1|0"
    AP = r"[a-zA-Z_][a-zA-Z0-9_]*"

    NOT = r"!"
    AND = r"&(&)?"
    OR = r"\|(\|)?"
    XOR = r"\^"
    IMPL = r"->"
    EQUIV = r"<->"

    # token remapping
    AP["X"] = NEXT
    AP["U"] = UNTIL
    AP["W"] = WUNTIL
    AP["R"] = RELEASE
    AP["F"] = EVEN
    AP["G"] = GLOB

    # @_(r'1|0')
    # def CONST(self, t):
    #     t.value = t.value == '1'
    #     return t

    def error(self, t):
        # TODO figure out how to return None instead of skipping illegal characters
        logger.debug(f"Illegal character {t.value[0]}")
        self.index += 1


# TODO straightforward inheritance from PrefixLexer does not seem possible
class LTLInfixLexer(sly.Lexer):

    bool_ops = {NOT, AND, OR, XOR, IMPL, EQUIV}
    temp_ops = {NEXT, UNTIL, WUNTIL, RELEASE, EVEN, GLOB}
    tokens = {AP, CONST, LPAR, RPAR}.union(bool_ops, temp_ops)

    ignore = " \t"

    CONST = r"true|false|1|0"
    AP = r"[a-zA-Z_][a-zA-Z0-9_]*"
    LPAR = r"\("
    RPAR = r"\)"

    NOT = r"!"
    AND = r"&(&)?"
    OR = r"\|(\|)?"
    XOR = r"\^"
    IMPL = r"->"
    EQUIV = r"<->"

    AP["X"] = NEXT
    AP["U"] = UNTIL
    AP["W"] = WUNTIL
    AP["R"] = RELEASE
    AP["F"] = EVEN
    AP["G"] = GLOB

    def error(self, t):
        logger.debug(f"Illegal character {t.value[0]}")
        self.index += 1


LTL_INFIX_LEXER = None


def lex_ltl(formula: str):
    global LTL_INFIX_LEXER
    if LTL_INFIX_LEXER is None:
        LTL_INFIX_LEXER = LTLInfixLexer()
    return [token.value for token in LTL_INFIX_LEXER.tokenize(formula)]
