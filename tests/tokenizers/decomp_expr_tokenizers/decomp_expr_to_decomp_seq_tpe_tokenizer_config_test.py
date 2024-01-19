"""Load ExprToSeqTPETokenizer test"""

from ml2.ltl import LTLGuarantees
from ml2.tokenizers.decomp_expr_tokenizers import DecompExprToDecompSeqTPETokenizer

DECOMP_EXPR_TO_DECOMP_SEQ_TPE_TOKENIZER_CONFIG = {
    "name": "decomp-expr-to-decomp-seq-tpe-tokenizer",
    "project": "test",
    "dtype": "LTLGuarantees",
    "num_sub_exprs": 4,
    "sub_tokenizer": {
        "type": "ExprToSeqTPETokenizer",
        "name": "decomp-expr-to-decomp-seq-tpe-tokenizer/sub-tokenizer",
        "project": "test",
        "dtype": "LTLSpec",
        "notation": "prefix",
        "pad": 6,
        "eos": False,
        "start": False,
        "tpe_format": "branch-up",
        "tpe_pad": 8,
        "vocabulary": {
            "name": "decomp-expr-to-decomp-seq-tpe-tokenizer/sub-tokenizer/vocabulary",
            "project": "test",
            "token_to_id": {
                "<p>": 0,
                "i1": 1,
                "i2": 2,
                "o1": 3,
                "o2": 4,
                "G": 5,
                "F": 6,
                "->": 7,
            },
        },
    },
}


def test_expr_to_seq_tpe_tokenizer_config():
    tokenizer = DecompExprToDecompSeqTPETokenizer.from_config(
        DECOMP_EXPR_TO_DECOMP_SEQ_TPE_TOKENIZER_CONFIG
    )
    guarantees = LTLGuarantees.from_dict(
        {
            "guarantees": ["G (i1 -> F o1)", "G (i2 -> F o2)"],
            "inputs": ["i1", "i2"],
            "outputs": ["o1", "o2"],
        }
    )
    encoding = tokenizer.encode(guarantees)
    assert len(encoding.seq_pos_encs) == 4
    assert encoding.seq_pos_encs[0].ids == [5, 7, 1, 6, 3, 0]
    assert encoding.seq_pos_encs[1].ids == [5, 7, 2, 6, 4, 0]
    assert encoding.seq_pos_encs[2].ids == [0, 0, 0, 0, 0, 0]
    assert encoding.seq_pos_encs[3].ids == [0, 0, 0, 0, 0, 0]
    assert encoding.seq_pos_encs[0].pad_pos_enc == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert encoding.seq_pos_encs[1].pad_pos_enc == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert encoding.seq_pos_encs[2].pad_pos_enc == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert encoding.seq_pos_encs[3].pad_pos_enc == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    config = tokenizer.get_config()
    assert config["name"] == "decomp-expr-to-decomp-seq-tpe-tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "LTLGuarantees"
    assert config["num_sub_exprs"] == 4
    assert config["sub_tokenizer"] == "test/decomp-expr-to-decomp-seq-tpe-tokenizer/sub-tokenizer"
    assert config["type"] == "DecompExprToDecompSeqTPETokenizer"
    sub_config = tokenizer.sub_tokenizer.get_config()
    assert sub_config["name"] == "decomp-expr-to-decomp-seq-tpe-tokenizer/sub-tokenizer"
    assert sub_config["project"] == "test"
    assert sub_config["dtype"] == "LTLSpec"
    assert sub_config["notation"] == "prefix"
    assert sub_config["pad"] == 6
    assert not sub_config["eos"]
    assert not sub_config["start"]
    assert sub_config["tpe_format"] == "branch-up"
    assert sub_config["tpe_pad"] == 8
    assert sub_config["type"] == "ExprToSeqTPETokenizer"
