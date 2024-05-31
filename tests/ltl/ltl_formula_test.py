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


def test_ltl_formula_default_precedence():
    formula_1 = LTLFormula.from_str(
        "(G((o1) -> (F((i1) -> (F(o2)))))) & (G((!(o1)) -> ((i1) <-> (!(o2))))) & ((F(o1)) -> ((i1) U (o1))) & ((o1) U ((i1) & (G(o2))))"
    )
    formula_2 = LTLFormula.from_str(formula_1.to_str(notation="infix-default-precedence"))
    assert formula_1 == formula_2


def test_ltl_formula_no_precedence():
    formula_1 = LTLFormula.from_str(
        "((F(i4)) -> ((i2) U (i4))) & (G((i4) -> (F((i2) & (X(F(o1))))))) & ((F(i4)) -> (((i2) -> ((!(i4)) U ((!(i4)) & (o1)))) U (i4))) & (X(G(((i4) & (!(i2))) -> ((G(!(i2))) | ((!(i2)) U (o1))))))"
    )
    formula_2 = LTLFormula.from_str(formula_1.to_str(notation="infix"))
    assert formula_1 == formula_2


def test_ltl_formula_precedence():
    precedence = [
        # low
        {"operator": ["<->", "->"]},
        {"assoc": "left", "operator": ["^"]},
        {"assoc": "left", "operator": ["|"]},
        {"assoc": "left", "operator": ["&"]},
        {"operator": ["U", "W", "R"]},
        {"assoc": "right", "operator": ["X", "!", "G"]},
        # high
    ]
    formula_1 = LTLFormula.from_str(
        "'(G((!(o4)) -> ((o1) <-> (!(i4))))) & (G(((o4) & (!(o1)) & (F(o1))) -> ((i4) U (o1))))'"
    )
    formula_2 = LTLFormula.from_str(formula_1.to_str(notation="infix", precedence=precedence))
    assert formula_1 == formula_2
