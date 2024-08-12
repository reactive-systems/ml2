"""LTL formula equiv test"""

from ml2.ltl.ltl_equiv import LTLEquivStatus, LTLInclStatus


def test_ltl_equiv_1():
    assert LTLEquivStatus("inequivalent").to_int() == 0
    assert LTLEquivStatus("error").to_int() == -1
    assert LTLEquivStatus("timeout").to_int() == -2
    assert LTLEquivStatus("equivalent").to_int() == 1
    assert LTLEquivStatus("equivalent").equiv


def test_ltl_equiv_2():
    try:
        LTLEquivStatus("satisfied")
        assert False
    except Exception as e:
        assert "invalid" in str(e).lower()


def test_ltl_incl_1():
    assert LTLInclStatus("incomparable").to_int() == 0
    assert LTLInclStatus("error").to_int() == -1
    assert LTLInclStatus("timeout").to_int() == -2
    assert LTLInclStatus("equivalent").to_int() == 1
    assert LTLInclStatus("equivalent").equiv
    assert LTLInclStatus("equivalent").left_in_right
    assert LTLInclStatus("equivalent").right_in_left
    assert LTLInclStatus("only_left_in_right").to_int() == 2
    assert LTLInclStatus("only_left_in_right").left_in_right
    assert LTLInclStatus("only_right_in_left").to_int() == 3
    assert LTLInclStatus("only_right_in_left").right_in_left


def test_ltl_incl_2():
    try:
        LTLInclStatus("inequivalent")
        assert False
    except Exception as e:
        assert "invalid" in str(e).lower()
