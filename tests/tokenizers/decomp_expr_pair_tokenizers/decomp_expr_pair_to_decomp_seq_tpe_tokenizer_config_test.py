"""Load ExprToSeqTPETokenizer test"""

from ml2.ltl import DecompLTLSpec
from ml2.tokenizers.decomp_expr_pair_tokenizers import DecompExprPairToDecompSeqTPETokenizer

DECOMP_EXPR_PAIR_TO_DECOMP_SEQ_TPE_TOKENIZER_CONFIG = {
    "name": "decomp-expr-pair-to-decomp-seq-tpe-tokenizer",
    "project": "test",
    "dtype": "DecompLTLSpec",
    "num_sub_exprs": 4,
    "sub_tokenizer": {
        "type": "ExprToSeqTPETokenizer",
        "name": "decomp-expr-pair-to-decomp-seq-tpe-tokenizer/sub-tokenizer",
        "project": "test",
        "dtype": "LTLSpec",
        "notation": "prefix",
        "pad": 6,
        "eos": False,
        "start": False,
        "tpe_format": "branch-up",
        "tpe_pad": 8,
        "vocabulary": {
            "name": "decomp-expr-pair-to-decomp-seq-tpe-tokenizer/sub-tokenizer/vocabulary",
            "project": "test",
            "token_to_id": {
                "<p>": 0,
                "<s>": 1,
                "i1": 2,
                "i2": 3,
                "o1": 4,
                "o2": 5,
                "G": 6,
                "F": 7,
                "->": 8,
            },
        },
    },
}


def test_expr_to_seq_tpe_tokenizer_config():
    tokenizer = DecompExprPairToDecompSeqTPETokenizer.from_config(
        DECOMP_EXPR_PAIR_TO_DECOMP_SEQ_TPE_TOKENIZER_CONFIG
    )
    spec = DecompLTLSpec.from_dict(
        {
            "assumptions": ["G F i1"],
            "guarantees": ["G (i1 -> F o1)", "G (i2 -> F o2)"],
            "inputs": ["i1", "i2"],
            "outputs": ["o1", "o2"],
        }
    )
    encoding = tokenizer.encode(spec)
    assert len(encoding.seq_pos_encs) == 4
    assert encoding.seq_pos_encs[0].ids == [1, 6, 7, 2, 0, 0]
    assert encoding.seq_pos_encs[1].ids == [6, 8, 2, 7, 4, 0]
    assert encoding.seq_pos_encs[2].ids == [6, 8, 3, 7, 5, 0]
    assert encoding.seq_pos_encs[3].ids == [0, 0, 0, 0, 0, 0]
    assert encoding.seq_pos_encs[0].pad_pos_enc == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
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
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
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
    assert config["name"] == "decomp-expr-pair-to-decomp-seq-tpe-tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "DecompLTLSpec"
    assert config["num_sub_exprs"] == 4
    assert (
        config["sub_tokenizer"]
        == "test/decomp-expr-pair-to-decomp-seq-tpe-tokenizer/sub-tokenizer"
    )
    assert config["type"] == "DecompExprPairToDecompSeqTPETokenizer"
    sub_config = tokenizer.sub_tokenizer.get_config()
    assert sub_config["name"] == "decomp-expr-pair-to-decomp-seq-tpe-tokenizer/sub-tokenizer"
    assert sub_config["project"] == "test"
    assert sub_config["dtype"] == "LTLSpec"
    assert sub_config["notation"] == "prefix"
    assert sub_config["pad"] == 6
    assert not sub_config["eos"]
    assert not sub_config["start"]
    assert sub_config["tpe_format"] == "branch-up"
    assert sub_config["tpe_pad"] == 8
    assert sub_config["type"] == "ExprToSeqTPETokenizer"
