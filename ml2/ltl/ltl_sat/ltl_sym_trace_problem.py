"""LTL symbolic trace generation problem"""

from typing import Dict, List

from ...dtypes import CSV, Supervised
from ...registry import register_type
from ...trace import SymbolicTrace
from ..ltl_formula import LTLFormula


@register_type
class LTLSymTraceProblem(CSV, Supervised[LTLFormula, SymbolicTrace]):
    def __init__(self, formula: LTLFormula, trace: SymbolicTrace) -> None:
        self.formula = formula
        self.trace = trace

    @property
    def input(self) -> LTLFormula:
        return self.formula

    @property
    def target(self) -> SymbolicTrace:
        return self.trace

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        formula_fields = self.formula.to_csv_fields(notation=notation, **kwargs)
        trace_fields = self.trace.to_csv_fields(**kwargs)
        return {**formula_fields, **trace_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(LTLFormula.csv_field_header(**kwargs) + SymbolicTrace.csv_field_header(**kwargs))
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSymTraceProblem":
        formula = LTLFormula.from_csv_fields(fields, **kwargs)
        trace = SymbolicTrace.from_csv_fields(fields, **kwargs)
        return cls(formula, trace)
