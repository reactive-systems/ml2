"""Propositional satisfiability problem in conjunctive normal form"""

from typing import Dict

from ...dtypes import CSV, Supervised
from ...registry import register_type
from ..prop_sat_status import PropSatStatus
from .cnf_formula import CNFFormula


class CNFSatSolution(PropSatStatus):
    def __init__(
        self,
        status: str,
        time: float = None,
        solver: str = None,
    ) -> None:
        self.time = time
        self.solver = solver
        super().__init__(status=status)

    def is_sat(self) -> bool:
        if self._status != "sat" and self._status != "unsat":
            raise Exception("SAT status not determined")
        return self._status == "sat"

    # TODO extend _to_csv_fields with time and solver fields

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFSatSolution":
        return super()._from_csv_fields(
            fields,
            time=fields.get("time", None),
            solver=fields.get("solver", None),
        )


@register_type
class CNFSatProblem(CSV, Supervised[CNFFormula, PropSatStatus]):
    def __init__(
        self,
        formula: CNFFormula,
        solution: CNFSatSolution = None,
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
    def target(self) -> CNFSatSolution:
        return self.solution

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        # TODO add id and timeout
        formula_csv_fields = self.formula.to_csv_fields(**kwargs)
        sol_csv_fields = self.solution.to_csv_fields(**kwargs) if self.solution is not None else {}
        return {**formula_csv_fields, **sol_csv_fields}

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFSatProblem":
        return cls(
            formula=CNFFormula.from_csv_fields(fields, **kwargs),
            solution=CNFSatSolution.from_csv_fields(fields) if "sat" in fields else None,
            id=fields.get("id"),
            timeout=fields.get("timeout"),
        )
