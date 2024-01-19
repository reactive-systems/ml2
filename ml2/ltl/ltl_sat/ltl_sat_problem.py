"""LTL satisfiability problem"""

from typing import Dict, List

from ...dtypes import CSV, Supervised
from ...registry import register_type
from ..ltl_formula import LTLFormula
from .ltl_sat_status import LTLSatStatus


@register_type
class LTLSatProblem(CSV, Supervised[LTLFormula, LTLSatStatus]):
    def __init__(self, formula: LTLFormula, status: LTLSatStatus) -> None:
        self.formula = formula
        self.status = status

    @property
    def input(self) -> LTLFormula:
        return self.formula

    @property
    def target(self) -> LTLSatStatus:
        return self.status

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        formula_fields = self.formula.to_csv_fields(notation=notation, **kwargs)
        status_fields = self.status.to_csv_fields(**kwargs)
        return {**formula_fields, **status_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(LTLFormula.csv_field_header(**kwargs) + LTLSatStatus.csv_field_header(**kwargs))
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSatProblem":
        formula = LTLFormula.from_csv_fields(fields, **kwargs)
        status = LTLSatStatus.from_csv_fields(fields, **kwargs)
        return cls(formula, status)
