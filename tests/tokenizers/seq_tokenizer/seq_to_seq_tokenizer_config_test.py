"""SeqToSeqTokenizer config test"""

from ml2.ltl import LTLFormula
from ml2.tokenizers import SeqToSeqTokenizer, Vocabulary

SEQ_TO_SEQ_TOKENIZER_CONFIG = {
    "dtype": "LTLFormula",
    "pad": 10,
    "eos": True,
    "start": True,
    "name": "seq-to-seq-tokenizer",
    "project": "test",
}

VOCAB_DICT = {
    "<p>": 0,
    "<s>": 1,
    "<e>": 2,
    "a": 3,
    "b": 4,
    "c": 5,
    "d": 6,
    "e": 7,
    "U": 8,
    "X": 9,
    "&": 10,
    "!": 11,
}


def test_seq_to_seq_tokenizer_config():
    vocabulary = Vocabulary(VOCAB_DICT)
    tokenizer = SeqToSeqTokenizer.from_config(SEQ_TO_SEQ_TOKENIZER_CONFIG, vocabulary=vocabulary)
    formula = LTLFormula.from_str("a U b & ! c")
    encoding = tokenizer.encode(formula)
    assert encoding.ids == [1, 3, 8, 4, 10, 11, 5, 2, 0, 0]
    config = tokenizer.get_config()
    assert config["dtype"] == "LTLFormula"
    assert config["pad"] == 10
    assert config["eos"]
    assert config["start"]
    assert config["name"] == "seq-to-seq-tokenizer"
    assert config["project"] == "test"
    assert config["type"] == "SeqToSeqTokenizer"
