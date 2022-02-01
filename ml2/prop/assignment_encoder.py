"""Assignment encoder"""

import logging
from itertools import chain

from ..data.encoder import SeqEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssignmentEncoder(SeqEncoder):
    def lex(self):
        self.tokens = list(chain.from_iterable((a[:-1], a[-1]) for a in self.sequence.split(",")))
        success = self.tokens is not None
        if not success:
            self.error = "Lex formula"
        return success

    @property
    def assignment(self):
        return {self.tokens[i]: int(self.tokens[i + 1]) for i in range(0, len(self.tokens), 2)}

    def vocabulary_filename(self):
        return "assignment-vocab" + super().vocabulary_filename()
