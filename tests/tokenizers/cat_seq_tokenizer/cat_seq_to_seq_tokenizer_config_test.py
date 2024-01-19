"""ExprToSeqTokenizer config test"""

from ml2.ltl.ltl_sat import LTLSatStatus, LTLSatTraceSolution
from ml2.tokenizers.cat_seq_tokenizers import CatSeqToSeqTokenizer
from ml2.trace import Trace

CAT_SEQ_TO_SEQ_TOKENIZER_CONFIG = {
    "name": "cat-seq-to-seq-tokenizer",
    "project": "test",
    "dtype": "LTLSatTraceSolution",
    "pad": 16,
    "eos": False,
    "start": False,
    "swap": True,
    "vocabulary": {
        "name": "cat-seq-to-seq-tokenizer/vocabulary",
        "project": "test",
        "token_to_id": {
            "<p>": 0,
            "satisfiable": 1,
            "unsatisfiable": 2,
            "a": 3,
            "b": 4,
            "{": 5,
            "}": 6,
            ";": 7,
        },
    },
}


def test_seq_to_seq_tokenizer_config():
    tokenizer = CatSeqToSeqTokenizer.from_config(CAT_SEQ_TO_SEQ_TOKENIZER_CONFIG)
    status = LTLSatStatus("satisfiable")
    trace = Trace.from_str("a ; { a ; b}")
    encoding = tokenizer.encode(LTLSatTraceSolution(status, trace))
    assert encoding.ids == [3, 7, 5, 3, 7, 4, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    config = tokenizer.get_config()
    assert config["name"] == "cat-seq-to-seq-tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "LTLSatTraceSolution"
    assert config["pad"] == 16
    assert not config["eos"]
    assert not config["start"]
    assert config["swap"]
    assert config["type"] == "CatSeqToSeqTokenizer"
