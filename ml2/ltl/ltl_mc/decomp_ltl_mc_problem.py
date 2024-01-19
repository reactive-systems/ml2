"""Decomposed LTL model checking problem"""


from typing import Dict, List, Optional, Tuple

from ...aiger import AIGERCircuit
from ...dtypes import CSV, Pair, Supervised
from ...registry import register_type
from ..ltl_spec import DecompLTLSpec
from ..ltl_syn import LTLRealStatus
from .ltl_mc_problem import LTLMCSolution


@register_type
class DecompLTLMCProblem(
    CSV,
    Supervised[Pair[DecompLTLSpec, Pair[LTLRealStatus, AIGERCircuit]], LTLMCSolution],
):
    def __init__(
        self,
        spec: DecompLTLSpec,
        circuit: AIGERCircuit,
        status: LTLRealStatus,
        solution: LTLMCSolution,
    ) -> None:
        self.spec = spec
        self.circuit = circuit
        self.status = status
        self.solution = solution

    @property
    def input(self) -> Tuple[DecompLTLSpec, AIGERCircuit, LTLRealStatus]:
        return (self.spec, self.circuit, self.status)

    @property
    def target(self) -> LTLMCSolution:
        return self.solution

    def _to_csv_fields(self, notation: Optional[str] = None, **kwargs) -> Dict[str, str]:
        spec_fields = self.spec.to_csv_fields(notation=notation, **kwargs)
        circuit_fields = self.circuit.to_csv_fields(**kwargs)
        status_fields = self.status.to_csv_fields(**kwargs)
        sol_fields = self.solution.to_csv_fields(**kwargs)
        return {**spec_fields, **circuit_fields, **status_fields, **sol_fields}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.spec)}\n{str(self.circuit)}\n{str(self.status)}\n{str(self.solution)}"

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(
                DecompLTLSpec.csv_field_header(**kwargs)
                + AIGERCircuit.csv_field_header(**kwargs)
                + LTLRealStatus.csv_field_header(**kwargs)
                + LTLMCSolution.csv_field_header(**kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "DecompLTLMCProblem":
        spec = DecompLTLSpec.from_csv_fields(fields, **kwargs)
        circuit = AIGERCircuit.from_csv_fields(fields, **kwargs)
        status = LTLRealStatus.from_csv_fields(fields, **kwargs)
        solution = LTLMCSolution.from_csv_fields(fields, **kwargs)
        return cls(spec, circuit, status, solution)
