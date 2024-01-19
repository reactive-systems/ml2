"""Propositional satisfiability problem"""

from typing import Dict, List

from ..dtypes import CSV, CatSeq, Supervised
from ..registry import register_type
from .assignment import Assignment
from .prop_formula import PropFormula
from .prop_sat_status import PropSatStatus


@register_type
class PropSatSolution(CSV, CatSeq[PropSatStatus, Assignment]):
    def __init__(
        self,
        status: PropSatStatus,
        assignment: Assignment,
    ):
        self.status = status
        self.assignment = assignment

    @property
    def cat(self) -> PropSatStatus:
        return self.status

    @property
    def seq(self) -> Assignment:
        return self.assignment

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        status_fields = self.status.to_csv_fields(**kwargs)
        assignment_fields = self.assignment.to_csv_fields(**kwargs)
        return {**status_fields, **assignment_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(PropSatStatus.csv_field_header(**kwargs) + Assignment.csv_field_header(**kwargs))
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "PropSatSolution":
        status = PropSatStatus.from_csv_fields(fields, **kwargs)
        assignment = Assignment.from_csv_fields(fields, **kwargs)
        return cls(
            status=status,
            assignment=assignment,
        )


@register_type
class PropSatProblem(CSV, Supervised[PropFormula, PropSatSolution]):
    def __init__(self, formula: PropFormula, solution: PropSatSolution) -> None:
        self.formula = formula
        self.solution = solution

    @property
    def input(self) -> PropFormula:
        return self.formula

    @property
    def target(self) -> PropSatSolution:
        return self.solution

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        formula_fields = self.formula.to_csv_fields(notation=notation, **kwargs)
        sol_fields = self.solution.to_csv_fields(**kwargs)
        return {**formula_fields, **sol_fields}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return list(
            set(
                PropFormula.csv_field_header(**kwargs) + PropSatSolution.csv_field_header(**kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "PropSatProblem":
        formula = PropFormula.from_csv_fields(fields, **kwargs)
        solution = PropSatSolution.from_csv_fields(fields, **kwargs)
        return cls(formula=formula, solution=solution)
