"""Aalta test"""

import pytest

from ml2.ltl import LTLFormula
from ml2.ltl.ltl_sat import LTLSatStatus
from ml2.tools.aalta import Aalta


@pytest.mark.docker
def test_aalta():
    aalta = Aalta()

    sat_formula = LTLFormula.from_str("a U b & G a")
    status, trace = aalta.check_sat(sat_formula)
    assert status == LTLSatStatus("satisfiable")

    unsat_formula = LTLFormula.from_str("a U b & G ! b")
    status, trace = aalta.check_sat(unsat_formula)
    assert status == LTLSatStatus("unsatisfiable")
