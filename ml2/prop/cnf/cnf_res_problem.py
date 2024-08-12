"""Propositional satisfiability problem in conjunctive normal form"""

from typing import Dict

from ...dtypes import CSV, Supervised
from ...registry import register_type
from .cnf_formula import CNFFormula
from .res_proof import ResProof


@register_type
class CNFResProblem(CSV, Supervised[CNFFormula, ResProof]):
    def __init__(
        self,
        formula: CNFFormula,
        proof: ResProof = None,
    ) -> None:
        self.formula = formula
        self.proof = proof

    @property
    def input(self) -> CNFFormula:
        return self.formula

    @property
    def target(self) -> ResProof:
        return self.proof

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        formula_csv_fields = self.formula.to_csv_fields(**kwargs)
        proof_csv_fields = self.proof.to_csv_fields(**kwargs) if self.proof is not None else {}
        return {**formula_csv_fields, **proof_csv_fields}

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFResProblem":
        return cls(
            formula=CNFFormula.from_csv_fields(fields, **kwargs),
            proof=ResProof.from_csv_fields(fields, **kwargs),
        )
