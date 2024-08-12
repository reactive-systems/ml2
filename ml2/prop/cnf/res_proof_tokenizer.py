"""Tokenizer that encodes a resolution proof into a sequence encoding"""

from typing import List, Type, TypeVar

from ...dtypes import Seq
from ...registry import register_type
from ...tokenizers.seq_tokenizers import SeqToSeqTokenizer
from ...tokenizers.tokenizer import TokenizationException
from .res_proof import ResProof

T = TypeVar("T", bound=Seq)


@register_type
class ResProofTokenizer(SeqToSeqTokenizer[ResProof]):
    def __init__(
        self,
        components: List[str] = None,
        dtype: Type[ResProof] = ResProof,
        pos_factor: int = 1,
        **kwargs,
    ):
        self.components = components if components is not None else ["id", "clause", "premises"]
        self.pos_factor = pos_factor

        super().__init__(dtype=dtype, **kwargs)

    def encode_tokens(self, data: ResProof, **kwargs) -> List[str]:
        int_tokens = []
        for rc in data.res_clauses:
            for c in self.components:
                if c == "id":
                    int_tokens.append(rc.id * self.pos_factor)
                elif c == "clause":
                    int_tokens.extend(rc.clause.lits)
                    int_tokens.append(0)
                elif c == "premises":
                    int_tokens.extend(rc.premises)
                    int_tokens.append(0)
                else:
                    raise TokenizationException(f"Unknown component {c}")
        return [str(t) for t in int_tokens]
