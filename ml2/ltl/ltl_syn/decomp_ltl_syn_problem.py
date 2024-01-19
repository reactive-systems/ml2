"""Decomposed LTL synthesis problem"""

import hashlib
from typing import Dict, List

from ...dtypes import CSVWithId, Supervised
from ...registry import register_type
from ..ltl_spec import DecompLTLSpec
from .ltl_syn_problem import LTLSynSolution


@register_type
class DecompLTLSynProblem(CSVWithId, Supervised[DecompLTLSpec, LTLSynSolution]):
    def __init__(self, ltl_spec: DecompLTLSpec, ltl_syn_solution=LTLSynSolution) -> None:
        self.ltl_spec = ltl_spec
        self.ltl_syn_solution = ltl_syn_solution

    @property
    def input(self) -> DecompLTLSpec:
        return self.ltl_spec

    @property
    def target(self) -> LTLSynSolution:
        return self.ltl_syn_solution

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        spec_fields = self.ltl_spec.to_csv_fields(notation=notation)
        sol_fields = self.ltl_syn_solution.to_csv_fields()
        return {**spec_fields, **sol_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(
                DecompLTLSpec.csv_field_header(**kwargs)
                + LTLSynSolution.csv_field_header(**kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "DecompLTLSynProblem":
        ltl_spec = DecompLTLSpec.from_csv_fields(fields, **kwargs)
        ltl_syn_solution = LTLSynSolution.from_csv_fields(fields, **kwargs)
        return cls(ltl_spec=ltl_spec, ltl_syn_solution=ltl_syn_solution)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str((self.ltl_spec.cr_hash, self.ltl_syn_solution.cr_hash)).encode()
            ).hexdigest(),
            16,
        )
