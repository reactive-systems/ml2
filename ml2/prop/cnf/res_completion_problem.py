from typing import Dict

from ...dtypes import CSV, Supervised
from ...registry import register_type
from .res_proof import ResProof


@register_type
class ResCompletionProblem(CSV, Supervised[ResProof, ResProof]):
    def __init__(
        self,
        proof_start: ResProof,
        proof_end: ResProof,
    ) -> None:
        self.proof_start = proof_start
        self.proof_end = proof_end

    @property
    def input(self) -> ResProof:
        return self.proof_start

    @property
    def target(self) -> ResProof:
        return self.proof_end

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {
            "proof_start": self.proof_start.to_csv_fields(**kwargs)["res_proof"],
            "proof_end": self.proof_end.to_csv_fields(**kwargs)["res_proof"],
        }

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "ResCompletionProblem":
        return cls(
            proof_start=ResProof.from_csv_fields({"res_proof": fields["proof_start"]}, **kwargs),
            proof_end=ResProof.from_csv_fields({"res_proof": fields["proof_end"]}, **kwargs),
        )
