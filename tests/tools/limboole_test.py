"""Limboole test"""

import pytest

from ml2.prop import PropSatStatus
from ml2.prop.prop_formula import PropFormula
from ml2.tools.limboole import Limboole


@pytest.mark.docker
def test_limboole():
    limboole = Limboole()

    sat_formula = PropFormula.from_str("! a & (a <-> b) & (c -> b)")
    status, assignment = limboole.check_sat(sat_formula)
    assert status == PropSatStatus("sat")

    unsat_formula = PropFormula.from_str("a & (a -> b) & (b <-> c) & (! c | ! a)")
    status, assignment = limboole.check_sat(unsat_formula)
    assert status == PropSatStatus("unsat")
