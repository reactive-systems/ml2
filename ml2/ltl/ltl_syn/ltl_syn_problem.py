"""LTL synthesis problem"""

import hashlib
from typing import Dict, List, Optional

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq, CSVWithId, Supervised
from ...registry import register_type
from ..ltl_spec import LTLSpec
from .ltl_syn_status import LTLSynStatus


@register_type
class LTLSynSolution(CSVWithId, CatSeq[LTLSynStatus, AIGERCircuit]):
    def __init__(
        self,
        status: LTLSynStatus,
        circuit: Optional[AIGERCircuit] = None,
        time: Optional[float] = None,
        tool: Optional[str] = None,
    ):
        self.status = status
        self.circuit = circuit
        self.time = time
        self.tool = tool

    @property
    def cat(self) -> LTLSynStatus:
        return self.status

    @property
    def seq(self) -> Optional[AIGERCircuit]:
        return self.circuit

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        if self.circuit is not None:
            circuit_fields = self.circuit.to_csv_fields(**kwargs)
        else:
            circuit_fields = {"circuit": ""}
        realizable_fields = self.status.to_csv_fields(**kwargs)
        time_fields = {} if self.time is None else {"syn_time": str(self.time)}
        tool_fields = {} if self.tool is None else {"syn_tool": self.tool}
        return {
            **circuit_fields,
            **realizable_fields,
            **time_fields,
            **tool_fields,
        }

    def to_str(self, **kwargs) -> str:
        return f"{self.status.token(**kwargs)} circuit {self.circuit.to_str(**kwargs)}"

    def __repr__(self):
        return f"{self.__class__.__name__}\nstatus: {str(self.status)}\ncircuit:\n{str(self.circuit)}\ntime: {str(self.time)}\ntool: {self.tool}"

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(
                AIGERCircuit.csv_field_header(**kwargs)
                + ["circuit"]
                + LTLSynStatus.csv_field_header(**kwargs)
                + ["syn_time", "syn_tool"]
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSynSolution":
        status: LTLSynStatus = LTLSynStatus.from_csv_fields(fields, **kwargs)  # type: ignore
        circuit: Optional[AIGERCircuit] = AIGERCircuit.from_csv_fields(fields, **kwargs)  # type: ignore
        time = (
            float(fields["syn_time"])
            if "syn_time" in fields and fields["syn_time"] != ""
            else None
        )
        tool = fields["syn_tool"] if "syn_tool" in fields else None
        return cls(status=status, circuit=circuit, time=time, tool=tool)

    @classmethod
    def from_cat_seq_pair(cls, cat: LTLSynStatus, seq: AIGERCircuit, **kwargs) -> "LTLSynSolution":
        return cls(status=cat, circuit=seq)

    @classmethod
    def from_cat_seq_tokens(
        cls, cat_token: str, seq_tokens: List[str], **kwargs
    ) -> "LTLSynSolution":
        return cls(
            status=LTLSynStatus.from_token(token=cat_token, **kwargs),
            circuit=AIGERCircuit.from_tokens(tokens=seq_tokens, **kwargs),
        )

    @classmethod
    def from_str(cls, s: str, **kwargs):
        status, circuit = s.split(" circuit ")
        return cls(
            status=LTLSynStatus.from_token(status, **kwargs),
            circuit=AIGERCircuit.from_str(circuit, **kwargs),
        )

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str((self.status._status, self.circuit.cr_hash if self.circuit else "")).encode()
            ).hexdigest(),
            16,
        )


@register_type
class LTLSynProblem(CSVWithId, Supervised[LTLSpec, LTLSynSolution]):
    def __init__(self, ltl_spec: LTLSpec, ltl_syn_solution: LTLSynSolution) -> None:
        self.ltl_spec = ltl_spec
        self.ltl_syn_solution = ltl_syn_solution

    @property
    def input(self) -> LTLSpec:
        return self.ltl_spec

    @property
    def target(self) -> LTLSynSolution:
        return self.ltl_syn_solution

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        spec_fields = self.ltl_spec.to_csv_fields(notation=notation, **kwargs)
        sol_fields = self.ltl_syn_solution.to_csv_fields(**kwargs)
        return {**spec_fields, **sol_fields}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.ltl_spec)}\n{str(self.ltl_syn_solution)}"

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(LTLSpec.csv_field_header(**kwargs) + LTLSynSolution.csv_field_header(**kwargs))
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSynProblem":
        ltl_spec = LTLSpec.from_csv_fields(fields, **kwargs)
        ltl_syn_solution = LTLSynSolution.from_csv_fields(fields, **kwargs)
        return cls(ltl_spec, ltl_syn_solution)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str((self.ltl_spec.cr_hash, self.ltl_syn_solution.cr_hash)).encode()
            ).hexdigest(),
            16,
        )
