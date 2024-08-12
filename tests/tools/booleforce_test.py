"""BooleForce test"""

import pytest

from ml2.prop import AssignmentCheckStatus
from ml2.prop.cnf import CNFAssignment, CNFFormula
from ml2.tools.booleforce import BooleForce

CSV_FIELDS = {
    "formula": "p cnf 10 49\\n3 2 9 0\\n-8 4 -9 -10 2 0\\n-9 -6 0\\n-9 1 10 0\\n6 4 -8 3 0\\n5 -10 2 6 0\\n-3 -1 -8 0\\n-8 -5 -2 9 0\\n3 7 -1 6 4 -8 10 5 9 0\\n-5 -4 2 0\\n-4 8 -6 2 0\\n5 -9 -8 0\\n4 -9 -8 -10 0\\n6 -2 9 0\\n10 -4 1 0\\n6 -5 -7 9 -10 -1 8 4 3 0\\n6 -8 9 10 -5 0\\n4 2 10 -3 0\\n8 6 7 0\\n-4 1 8 0\\n10 9 -1 -2 0\\n-5 -1 3 0\\n5 -8 6 -7 0\\n-6 -1 0\\n10 9 0\\n-8 9 4 0\\n2 -6 8 0\\n7 8 1 0\\n-2 8 10 1 0\\n-8 7 -10 0\\n6 1 -10 7 4 0\\n-10 4 -7 0\\n-8 -7 3 0\\n7 4 3 9 10 -8 2 -1 6 -5 0\\n-9 6 4 8 0\\n2 7 -10 0\\n-3 5 -2 0\\n7 -10 -2 4 0\\n-4 -2 10 -8 0\\n-9 4 -3 -7 1 -8 -5 0\\n-4 6 3 0\\n-6 -7 0\\n10 -1 5 -7 -3 0\\n-1 10 0\\n-1 -3 -10 -7 0\\n2 6 5 -4 0\\n4 -1 -8 6 0\\n-3 2 -5 0\\n-7 -2 -10 0\\n"
}


@pytest.mark.docker
def test_booleforce():
    formula = CNFFormula.from_csv_fields(CSV_FIELDS)

    booleforce = BooleForce()
    sol = booleforce.check_sat(formula)
    assert not sol.is_sat()

    check = booleforce.trace_check(sol.res_proof)
    assert check.token() == "resolved"

    bin_proof = booleforce.binarize_res_proof(sol.res_proof)
    for rc in bin_proof.res_clauses:
        assert len(rc.premises) <= 2

    ps = bin_proof.to_str(notation="tracecheck-sorted")
    cs = ps.strip().split("\n")
    assert sorted(cs, key=lambda x: int(x.split()[0])) == cs


@pytest.mark.docker
def test_booleforce_check_asssignment():
    booleforce = BooleForce()
    formula = CNFFormula.from_int_lists([[1, 2], [-1]])
    sat_assign = CNFAssignment([-1, 2])
    unsat_assign_1 = CNFAssignment([-1, -2])
    unsat_assign_2 = CNFAssignment([1, 2])
    unsat_assign_3 = CNFAssignment([1, -2])
    sat_status = AssignmentCheckStatus("satisfying")
    unsat_status = AssignmentCheckStatus("unsatisfying")
    assert booleforce.check_assignment(formula, sat_assign) == sat_status
    assert booleforce.check_assignment(formula, unsat_assign_1) == unsat_status
    assert booleforce.check_assignment(formula, unsat_assign_2) == unsat_status
    assert booleforce.check_assignment(formula, unsat_assign_3) == unsat_status
