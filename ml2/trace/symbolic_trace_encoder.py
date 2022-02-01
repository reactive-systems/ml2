"""Symbolic trace encoder"""

import logging

from .symbolic_trace import SymbolicTrace
from ..data.encoder import SeqEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolicTraceEncoder(SeqEncoder):
    def __init__(self, notation: str, encoded_notation: str, **kwargs) -> None:
        self.notation = notation
        self.encoded_notation = encoded_notation
        super().__init__(**kwargs)

    def lex(self) -> bool:
        self.tokens = SymbolicTrace.from_str(trace=self.sequence, notation=self.notation).tokens(
            self.encoded_notation
        )
        success = self.tokens is not None
        if not success:
            self.error = "Lex formula"
        return success

    def detokenize(self) -> bool:
        self.sequence = SymbolicTrace.from_tokens(
            self.tokens, notation=self.encoded_notation
        ).to_str(notation=self.notation)
        return self.sequence is not None

    def vocabulary_filename(self) -> str:
        return "symbolic-trace-vocab" + super().vocabulary_filename()
