"""LTL encoder"""

import logging

from ..data.encoder import ExprEncoder, SeqEncoder
from ..data.expr import ExprNotation
from .ltl_lexer import lex_ltl
from .ltl_parser import parse_prefix_ltl, parse_infix_ltl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSequenceEncoder(SeqEncoder):
    @property
    def formula(self):
        return self.sequence

    def lex(self) -> bool:
        self.tokens = lex_ltl(self.formula)
        success = self.tokens is not None
        if not success:
            self.error = "Lex formula"
        return success

    def vocabulary_filename(self) -> str:
        return "ltl-vocab" + super().vocabulary_filename()


class LTLTreeEncoder(ExprEncoder):
    @property
    def formula(self):
        return self.expression

    def lex(self) -> bool:
        self.tokens = lex_ltl(self.formula)
        success = self.tokens is not None
        if not success:
            self.error = "Lex formula"
        return success

    def parse(self) -> bool:
        if self.notation == ExprNotation.PREFIX:
            self.ast = parse_prefix_ltl(self.formula)
        elif self.notation == ExprNotation.INFIX:
            self.ast = parse_infix_ltl(self.formula)
        else:
            logger.critical("Unsupported notation %s", self.notation)
        success = self.ast is not None
        if not success:
            self.error = "Parse formula"
        return success

    def vocabulary_filename(self) -> str:
        return "ltl-vocab" + super().vocabulary_filename()
