"""Load ExprToSeqTPETokenizer test"""

from ml2.ltl import LTLFormula
from ml2.tokenizers.expr_tokenizers import ExprToSeqTPETokenizer

EXPR_TO_SEQ_TPE_TOKENIZER_CONFIG = {
    "name": "expr-to-seq-tpe-tokenizer",
    "project": "test",
    "dtype": "LTLFormula",
    "notation": "prefix",
    "pad": 8,
    "eos": False,
    "start": False,
    "tpe_format": "branch-up",
    "tpe_pad": 10,
    "vocabulary": {
        "name": "expr-to-seq-tpe-tokenizer/vocabulary",
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


def test_expr_to_seq_tpe_tokenizer_config():
    tokenizer = ExprToSeqTPETokenizer.from_config(EXPR_TO_SEQ_TPE_TOKENIZER_CONFIG)
    formula = LTLFormula.from_str("a U b & ! c")
    encoding = tokenizer.encode(formula)
    assert encoding.ids == [10, 8, 3, 4, 11, 5, 0, 0]
    assert encoding.pad_pos_enc == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    config = tokenizer.get_config()
    assert config["name"] == "expr-to-seq-tpe-tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "LTLFormula"
    assert config["notation"] == "prefix"
    assert config["pad"] == 8
    assert not config["eos"]
    assert not config["start"]
    assert config["tpe_format"] == "branch-up"
    assert config["tpe_pad"] == 10
    assert config["type"] == "ExprToSeqTPETokenizer"
