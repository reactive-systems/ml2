"""LTL model checking problem"""

import hashlib
from typing import Dict, List, Optional, Tuple

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq, CSVLoggableValidationResult, CSVWithId, Pair, Supervised
from ...trace import Trace
from ..ltl_spec import LTLSpec
from ..ltl_syn import LTLRealStatus
from .ltl_mc_status import LTLMCStatus


class LTLMCSolution(CSVWithId, CSVLoggableValidationResult, CatSeq[LTLMCStatus, Trace]):
    def __init__(
        self,
        status: LTLMCStatus,
        trace: Optional[Trace] = None,
        time: Optional[float] = None,
        tool: Optional[str] = None,
    ):
        self.status = status
        self.trace = trace
        self.time = time
        self.tool = tool

    @property
    def cat(self) -> LTLMCStatus:
        return self.status

    @property
    def validation_success(self) -> Optional[bool]:
        if self.status is not None:
            return self.status.satisfied
        else:
            return None

    @property
    def validation_status(self) -> Optional[str]:
        if self.status is not None:
            return self.status.token()
        else:
            return None

    @property
    def value(self) -> str:
        return self.status.value

    @property
    def seq(self) -> Optional[Trace]:
        return self.trace

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        if self.trace is not None:
            trace_fields = self.trace.to_csv_fields(**kwargs)
        else:
            trace_fields = {"trace": ""}
        result_fields = self.status.to_csv_fields(**kwargs)
        time_fields = {} if self.time is None else {"mc_time": str(self.time)}
        tool_fields = {} if self.tool is None else {"mc_tool": self.tool}
        return {
            **trace_fields,
            **result_fields,
            **time_fields,
            **tool_fields,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}\nstatus: {str(self.status)}\ntrace:\n{None if self.trace is None else str(self.trace.to_str())}\ntime: {str(self.time)}\ntool: {self.tool}"

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(
                Trace.csv_field_header(**kwargs)
                + ["trace"]
                + LTLMCStatus.csv_field_header(**kwargs)
                + ["mc_time", "mc_tool"]
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLMCSolution":
        status = LTLMCStatus.from_csv_fields(fields, **kwargs)  # type: ignore
        trace = Trace.from_csv_fields(fields, **kwargs)  # type: ignore
        time = float(fields["mc_time"]) if "mc_time" in fields else None
        tool = fields["mc_tool"] if "mc_tool" in fields else None
        return cls(status=status, trace=trace, time=time, tool=tool)

    @classmethod
    def from_cat_seq_pair(cls, cat: LTLMCStatus, seq: Trace, **kwargs) -> "LTLMCSolution":
        return cls(status=cat, trace=seq)

    @classmethod
    def from_cat_seq_tokens(
        cls, cat_token: str, seq_tokens: List[str], **kwargs
    ) -> "LTLMCSolution":
        return cls(
            status=LTLMCStatus.from_token(token=cat_token, **kwargs),
            trace=Trace.from_tokens(tokens=seq_tokens, **kwargs),
        )

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str((self.status._status, self.trace.cr_hash if self.trace else "")).encode()
            ).hexdigest(),
            16,
        )


class LTLMCProblem(
    CSVWithId,
    Supervised[Pair[LTLSpec, Pair[LTLRealStatus, AIGERCircuit]], LTLMCSolution],
):
    def __init__(
        self,
        spec: LTLSpec,
        circuit: AIGERCircuit,
        status: LTLRealStatus,
        solution: LTLMCSolution,
    ) -> None:
        self.spec = spec
        self.circuit = circuit
        self.status = status
        self.solution = solution

    @property
    def input(self) -> Tuple[LTLSpec, AIGERCircuit, LTLRealStatus]:
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
                LTLSpec.csv_field_header(**kwargs)
                + AIGERCircuit.csv_field_header(**kwargs)
                + LTLRealStatus.csv_field_header(**kwargs)
                + LTLMCSolution.csv_field_header(**kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLMCProblem":
        spec = LTLSpec.from_csv_fields(fields, **kwargs)
        circuit = AIGERCircuit.from_csv_fields(fields, **kwargs)
        status = LTLRealStatus.from_csv_fields(fields, **kwargs)
        solution = LTLMCSolution.from_csv_fields(fields, **kwargs)
        return cls(spec, circuit, status, solution)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.spec.cr_hash,
                        self.circuit.cr_hash,
                        self.status,
                        self.solution.cr_hash,
                    )
                ).encode()
            ).hexdigest(),
            16,
        )
