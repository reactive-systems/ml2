"""LTL satisfiability and trace problem"""

from typing import Dict, List

from ...dtypes import CSV, CatSeq, Supervised
from ...registry import register_type
from ...trace import Trace
from ..ltl_formula import LTLFormula
from .ltl_sat_status import LTLSatStatus


@register_type
class LTLSatTraceSolution(CSV, CatSeq[LTLSatStatus, Trace]):
    def __init__(self, status: LTLSatStatus, trace: Trace):
        self.status = status
        self.trace = trace

    @property
    def cat(self) -> LTLSatStatus:
        return self.status

    @property
    def seq(self) -> Trace:
        return self.trace

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        status_fields = self.status.to_csv_fields(**kwargs)
        trace_fields = self.trace.to_csv_fields(**kwargs)
        return {**status_fields, **trace_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(LTLSatStatus.csv_field_header(**kwargs) + Trace.csv_field_header(**kwargs))
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSatTraceSolution":
        status = LTLSatStatus.from_csv_fields(fields, **kwargs)
        trace = Trace.from_csv_fields(fields, **kwargs)
        return cls(status, trace)

    @classmethod
    def from_cat_seq_pair(cls, cat: LTLSatStatus, seq: Trace, **kwargs) -> "LTLSatTraceSolution":
        return cls(status=cat, trace=seq)

    @classmethod
    def from_cat_seq_tokens(
        cls, cat_token: str, seq_tokens: List[str], **kwargs
    ) -> "LTLSatTraceSolution":
        return cls(
            status=LTLSatStatus.from_token(token=cat_token, **kwargs),
            trace=Trace.from_tokens(tokens=seq_tokens, **kwargs),
        )


@register_type
class LTLSatTraceProblem(CSV, Supervised[LTLFormula, LTLSatTraceSolution]):
    def __init__(self, formula: LTLFormula, solution: LTLSatTraceSolution) -> None:
        self.formula = formula
        self.solution = solution

    @property
    def input(self) -> LTLFormula:
        return self.formula

    @property
    def target(self) -> LTLSatTraceSolution:
        return self.solution

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        formula_fields = self.formula.to_csv_fields(notation=notation, **kwargs)
        sol_fields = self.solution.to_csv_fields(**kwargs)
        return {**formula_fields, **sol_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(
                LTLFormula.csv_field_header(**kwargs)
                + LTLSatTraceSolution.csv_field_header(**kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSatTraceProblem":
        formula = LTLFormula.from_csv_fields(fields, **kwargs)
        solution = LTLSatTraceSolution.from_csv_fields(fields, **kwargs)
        return cls(formula, solution)
