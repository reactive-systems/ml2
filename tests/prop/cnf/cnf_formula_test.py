"""CNF formula test"""

from pathlib import Path

from ml2.prop.cnf.cnf_formula import CNFFormula

CSV_FIELDS = {
    "formula": "p cnf 10 49\\n3 2 9 0\\n-8 4 -9 -10 2 0\\n-9 -6 0\\n-9 1 10 0\\n6 4 -8 3 0\\n5 -10 2 6 0\\n-3 -1 -8 0\\n-8 -5 -2 9 0\\n3 7 -1 6 4 -8 10 5 9 0\\n-5 -4 2 0\\n-4 8 -6 2 0\\n5 -9 -8 0\\n4 -9 -8 -10 0\\n6 -2 9 0\\n10 -4 1 0\\n6 -5 -7 9 -10 -1 8 4 3 0\\n6 -8 9 10 -5 0\\n4 2 10 -3 0\\n8 6 7 0\\n-4 1 8 0\\n10 9 -1 -2 0\\n-5 -1 3 0\\n5 -8 6 -7 0\\n-6 -1 0\\n10 9 0\\n-8 9 4 0\\n2 -6 8 0\\n7 8 1 0\\n-2 8 10 1 0\\n-8 7 -10 0\\n6 1 -10 7 4 0\\n-10 4 -7 0\\n-8 -7 3 0\\n7 4 3 9 10 -8 2 -1 6 -5 0\\n-9 6 4 8 0\\n2 7 -10 0\\n-3 5 -2 0\\n7 -10 -2 4 0\\n-4 -2 10 -8 0\\n-9 4 -3 -7 1 -8 -5 0\\n-4 6 3 0\\n-6 -7 0\\n10 -1 5 -7 -3 0\\n-1 10 0\\n-1 -3 -10 -7 0\\n2 6 5 -4 0\\n4 -1 -8 6 0\\n-3 2 -5 0\\n-7 -2 -10 0\\n"
}

DIMACS_FILEPATH = Path(__file__).parent / "formula.dimacs"


def test_cnf_formula():
    formula = CNFFormula.from_csv_fields(CSV_FIELDS)
    assert formula.num_clauses == 49
    assert formula.num_vars == 10
    assert formula.to_csv_fields() == CSV_FIELDS


def test_cnf_formula_from_dimacs_file():
    formula = CNFFormula.from_dimacs_file(DIMACS_FILEPATH)
    assert formula.num_clauses == 7
    assert formula.num_vars == 5
    with open(DIMACS_FILEPATH, "r") as file:
        assert formula.to_str(notation="dimacs") == file.read()
