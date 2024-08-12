"""Tokenizer that encodes a CNF Formula into a sequence encoding"""

import itertools
from typing import List, Type, TypeVar

from ...dtypes import Seq
from ...registry import register_type
from ...tokenizers.seq_tokenizers import SeqToSeqTokenizer
from .cnf_formula import CNFFormula

T = TypeVar("T", bound=Seq)


@register_type
class CNFFormulaTokenizer(SeqToSeqTokenizer[CNFFormula]):
    def __init__(
        self,
        dtype: Type[CNFFormula] = CNFFormula,
        enumerated: bool = False,
        pos_factor: int = 1,
        **kwargs,
    ):
        self.enumerated = enumerated
        self.pos_factor = pos_factor
        super().__init__(dtype=dtype, **kwargs)

    def encode_tokens(self, data: CNFFormula, **kwargs) -> List[str]:
        nested_tokens = []
        for i, clause in enumerate(data.clauses):
            clause_tokens = clause.tokens(**kwargs)
            if self.enumerated:
                pos = (i + 1) * self.pos_factor
                clause_tokens = [str(pos)] + clause_tokens
            nested_tokens.append(clause_tokens)
        return list(itertools.chain.from_iterable(nested_tokens))
