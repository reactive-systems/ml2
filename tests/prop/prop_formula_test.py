"""Propositional formula test"""

from ml2.prop.prop_formula import PropFormula

FORMULA_STR = "a <-> (! b & c)"


def test_prop_formula():
    formula = PropFormula.from_str(FORMULA_STR)
    assert formula.to_str() == FORMULA_STR
    assert formula.to_str(notation="prefix") == "<-> a & ! b c"
