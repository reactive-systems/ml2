"""ExprToSeqTokenizer config test"""

from ml2.ltl import LTLFormula
from ml2.tokenizers import ExprToSeqTokenizer

EXPR_TO_SEQ_TOKENIZER_CONFIG = {
    "name": "expr-to-seq-tokenizer",
    "project": "test",
    "dtype": "LTLFormula",
    "notation": "prefix",
    "pad": 16,
    "eos": False,
    "start": False,
    "vocabulary": {
        "name": "expr-to-seq-tokenizer/vocabulary",
        "project": "test",
        "token_to_id": {
            "<p>": 0,
            "a": 3,
            "b": 4,
            "c": 5,
            "d": 6,
            "e": 7,
            "U": 8,
            "X": 9,
            "&": 10,
            "!": 11,
        },
    },
}


def test_expr_to_seq_tokenizer_config():
    tokenizer = ExprToSeqTokenizer.from_config(EXPR_TO_SEQ_TOKENIZER_CONFIG)
    formula = LTLFormula.from_str("a U b & ! c")
    encoding = tokenizer.encode(formula)
    assert encoding.ids == [10, 8, 3, 4, 11, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    config = tokenizer.get_config()
    assert config["name"] == "expr-to-seq-tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "LTLFormula"
    assert config["notation"] == "prefix"
    assert config["pad"] == 16
    assert not config["eos"]
    assert not config["start"]
    assert config["type"] == "ExprToSeqTokenizer"
