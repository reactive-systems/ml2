"""Symbolic trace to sequence tokenizer config test"""

from ml2.tokenizers import Vocabulary
from ml2.trace import SymbolicTrace, SymTraceToSeqTokenizer

SYM_TRACE_TO_SEQ_TOKENIZER_CONFIG = {
    "notation": "prefix",
    "name": "sym-trace-to-seq-tokenizer",
    "project": "test",
}


def test_sym_trace_to_seq_tokenizer_config():
    vocabulary = Vocabulary(
        token_to_id={}, name="sym-trace-to-seq-tokenizer/vocabulary", project="test"
    )
    tokenizer = SymTraceToSeqTokenizer.from_config(
        SYM_TRACE_TO_SEQ_TOKENIZER_CONFIG, vocabulary=vocabulary
    )
    assert tokenizer.dtype == SymbolicTrace
    assert tokenizer.notation == "prefix"
    config = tokenizer.get_config()
    assert "notation" in config and config["notation"] == "prefix"
