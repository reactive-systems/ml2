"""Common functionality for resolution data generation"""

import random
from copy import deepcopy
from typing import Dict

import numpy as np

from .cnf_formula import Clause, CNFFormula
from .cnf_sat_search_problem import CNFSatSearchProblem


class CNFResDataGenProblem(CNFSatSearchProblem):
    def add_clause(self, p_k_2: float, p_geo: float) -> "CNFResDataGenProblem":
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        num_vars = self.formula.num_vars
        vars = np.random.choice(num_vars, size=min(num_vars, k), replace=False)
        lits = [v + 1 if random.random() < 0.5 else -v - 1 for v in vars]
        clause = Clause.from_list(lits)
        formula = deepcopy(self.formula)
        formula.add_clause(clause)
        return CNFResDataGenProblem(formula=formula, id=self.id, timeout=self.timeout)

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        formula_csv_fields = self.formula.to_csv_fields(**kwargs)
        proof_csv_fields = (
            self.solution.res_proof.to_csv_fields(**kwargs)
            if self.solution.res_proof is not None
            else {}
        )
        return {**formula_csv_fields, **proof_csv_fields}

    @classmethod
    def from_random(cls, min_n: int, max_n: int, **kwargs) -> "CNFResDataGenProblem":
        num_vars = random.randint(min_n, max_n)
        formula = CNFFormula(num_vars=num_vars)
        return cls(formula=formula, **kwargs)
