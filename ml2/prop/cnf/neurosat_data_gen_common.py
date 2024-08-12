"""Common functionality for NeuroSAT data generation"""

import random
from copy import deepcopy
from typing import Dict

import numpy as np

from .cnf_formula import Clause, CNFFormula
from .cnf_sat_search_problem import CNFSatSearchProblem, CNFSatSearchSolution


class NeuroSatProblem(CNFSatSearchProblem):
    def __init__(
        self,
        formula: CNFFormula,
        solution: CNFSatSearchSolution = None,
        id: int = None,
        parent: "NeuroSatProblem" = None,
        timeout: float = None,
    ):
        super().__init__(formula=formula, solution=solution, id=id, timeout=timeout)
        self.parent = parent

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        if self.solution is None:
            raise ValueError("Solution is None")
        return super()._to_csv_fields(**kwargs)

    def add_clause(self, p_k_2: float, p_geo: float) -> "NeuroSatProblem":
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        num_vars = self.formula.num_vars
        vars = np.random.choice(num_vars, size=min(num_vars, k), replace=False)
        lits = [v + 1 if random.random() < 0.5 else -v - 1 for v in vars]
        clause = Clause.from_list(lits)
        formula = deepcopy(self.formula)
        formula.add_clause(clause)
        self.parent = None
        return NeuroSatProblem(formula=formula, id=self.id, timeout=self.timeout, parent=self)

    @classmethod
    def from_random(cls, min_n: int, max_n: int, **kwargs) -> "NeuroSatProblem":
        num_vars = random.randint(min_n, max_n)
        formula = CNFFormula(num_vars=num_vars)
        return cls(formula=formula, **kwargs)


def add_neurosat_data_gen_args(parser):
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--no-shuffle", action="store_false", help="dataset not shuffled", dest="shuffle"
    )
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=40)
    parser.add_argument("--p-k-2", type=float, default=0.3)
    parser.add_argument("--p-geo", type=float, default=0.4)
