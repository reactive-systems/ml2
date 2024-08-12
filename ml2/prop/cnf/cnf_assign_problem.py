"""Satisfiable propositional problem in conjunctive normal form asking for sat assignment"""

from typing import Dict

from ...dtypes import CSV, Supervised
from ...registry import register_type
from .cnf_assignment import CNFAssignment
from .cnf_formula import CNFFormula


@register_type
class CNFAssignProblem(CSV, Supervised[CNFFormula, CNFAssignment]):

    def __init__(
        self,
        formula: CNFFormula,
        solution: CNFAssignment = None,
        id: int = None,
        timeout: float = None,
    ) -> None:
        self.formula = formula
        self.solution = solution
        self.id = id
        self.timeout = timeout

    @property
    def input(self) -> CNFFormula:
        return self.formula

    @property
    def target(self) -> CNFAssignment:
        return self.solution

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        # TODO add id and timeout
        formula_csv_fields = self.formula.to_csv_fields(**kwargs)
        sol_csv_fields = self.solution.to_csv_fields(**kwargs) if self.solution is not None else {}
        return {**formula_csv_fields, **sol_csv_fields}

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFAssignProblem":
        return cls(
            formula=CNFFormula.from_csv_fields(fields, **kwargs),
            solution=CNFAssignment.from_csv_fields(fields),
            id=fields.get("id"),
            timeout=fields.get("timeout"),
        )
