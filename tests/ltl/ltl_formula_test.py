"""LTL formula test"""

from ml2.dtypes import BinaryAST
from ml2.ltl import LTLFormula


def test_ltl_formula_str_infix():
    formula = LTLFormula.from_str("a U (b & X c)")
    assert formula.ast == BinaryAST(
        "U", BinaryAST("a"), BinaryAST("&", BinaryAST("b"), BinaryAST("X", BinaryAST("c")))
    )
    assert formula.size() == 6
    assert formula.to_tokens() == ["a", "U", "(", "b", "&", "X", "c", ")"]
    assert formula.to_str() == "a U (b & X c)"
    assert formula.to_tokens(notation="prefix") == ["U", "a", "&", "b", "X", "c"]
    assert formula.to_str(notation="prefix") == "U a & b X c"


def test_ltl_formula_str_prefix():
    formula = LTLFormula.from_str("& ! a U a X b", notation="prefix")
    assert formula.ast == BinaryAST(
        "&",
        BinaryAST("!", BinaryAST("a")),
        BinaryAST("U", BinaryAST("a"), BinaryAST("X", BinaryAST("b"))),
    )
    assert formula.size() == 7
    assert formula.to_tokens() == ["&", "!", "a", "U", "a", "X", "b"]
    assert formula.to_str() == "& ! a U a X b"
    assert formula.to_tokens(notation="infix") == [
        "(",
        "!",
        "(",
        "a",
        ")",
        ")",
        "&",
        "(",
        "(",
        "a",
        ")",
        "U",
        "(",
        "X",
        "(",
        "b",
        ")",
        ")",
        ")",
    ]
    assert formula.to_str(notation="infix") == "( ! ( a ) ) & ( ( a ) U ( X ( b ) ) )"
