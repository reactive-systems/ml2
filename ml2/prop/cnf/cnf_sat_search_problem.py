"""Propositional satisfiability problem in conjunctive normal form asking for either sat assignment or unsat proof"""

from typing import Dict

from ...dtypes import CSV
from ...registry import register_type
from ..prop_sat_status import PropSatStatus
from .cnf_assignment import CNFAssignment
from .cnf_formula import CNFFormula
from .res_proof import ResProof


class CNFSatSearchSolution(CSV):
    def __init__(
        self,
        status: PropSatStatus,
        assignment: CNFAssignment = None,
        res_proof: ResProof = None,
        clausal_proof=None,
        time: float = None,
        solver: str = None,
    ) -> None:
        self.status = status
        self.assignment = assignment
        self.res_proof = res_proof
        self.clausal_proof = clausal_proof
        self.time = time
        self.solver = solver
        super().__init__()

    def is_sat(self) -> bool:
        if self.status._status != "sat" and self.status._status != "unsat":
            raise Exception("SAT status not determined")
        return self.status._status == "sat"

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        # TODO clausal proof string, time and solver fields
        status_fields = self.status.to_csv_fields(**kwargs)
        assignment_fields = (
            self.assignment.to_csv_fields(**kwargs) if self.assignment is not None else {}
        )
        res_proof_fields = (
            self.res_proof.to_csv_fields(**kwargs) if self.res_proof is not None else {}
        )
        clausal_proof_fields = {}
        return {
            **status_fields,
            **assignment_fields,
            **res_proof_fields,
            **clausal_proof_fields,
        }

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFSatSearchSolution":
        return cls(
            status=PropSatStatus.from_csv_fields(fields),
            assignment=(
                CNFAssignment.from_csv_fields(fields)
                if "assignment" in fields and fields["assignment"] != ""
                else None
            ),
            res_proof=(
                ResProof.from_csv_fields(fields)
                if "res_proof" in fields and fields["res_proof"] != ""
                else None
            ),
            clausal_proof=None,
            time=fields.get("time", None),
            solver=fields.get("solver", None),
        )


@register_type
class CNFSatSearchProblem(CSV):
    def __init__(
        self,
        formula: CNFFormula,
        solution: CNFSatSearchSolution = None,
        id: int = None,
        timeout: float = None,
    ) -> None:
        self.formula = formula
        self.solution = solution
        self.id = id
        self.timeout = timeout

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        # TODO add id and timeout
        formula_csv_fields = self.formula.to_csv_fields(**kwargs)
        sol_csv_fields = self.solution.to_csv_fields(**kwargs) if self.solution is not None else {}
        return {**formula_csv_fields, **sol_csv_fields}

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFSatSearchProblem":
        return cls(
            formula=CNFFormula.from_csv_fields(fields, **kwargs),
            solution=CNFSatSearchSolution.from_csv_fields(fields) if "sat" in fields else None,
            id=fields.get("id"),
            timeout=fields.get("timeout"),
        )
