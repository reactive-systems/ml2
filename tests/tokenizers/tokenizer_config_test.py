"""Tokenizer config test"""

import numpy as np
import tensorflow as tf
import torch

from ml2.ltl import LTLFormula
from ml2.tokenizers import Tokenizer

VOCABULARY_CONFIG = {
    "name": "tokenizer/vocabulary",
    "project": "test",
    "token_to_id": {"<p>": 0, "a": 1, "b": 2, "U": 3},
}

TOKENIZER_CONFIG = {
    "name": "tokenizer",
    "project": "test",
    "dtype": "LTLFormula",
    "np_int_dtype": "int16",
    "pt_int_dtype": "int16",
    "tf_int_dtype": "int16",
    "vocabulary": VOCABULARY_CONFIG,
}


def test_tokenizer_config():
    tokenizer = Tokenizer.from_config(TOKENIZER_CONFIG)
    assert tokenizer.name == "tokenizer"
    assert tokenizer.project == "test"
    assert tokenizer.dtype == LTLFormula
    assert tokenizer.np_int_dtype == np.int16
    assert tokenizer.pt_int_dtype == torch.int16
    assert tokenizer.tf_int_dtype == tf.int16
    vocabulary = tokenizer.vocabulary
    assert vocabulary.name == "tokenizer/vocabulary"
    assert vocabulary.project == "test"
    assert vocabulary.token_to_id == {"<p>": 0, "a": 1, "b": 2, "U": 3}
    config = tokenizer.get_config()
    assert config["name"] == "tokenizer"
    assert config["project"] == "test"
    assert config["dtype"] == "LTLFormula"
    assert config["np_int_dtype"] == "int16"
    assert config["pt_int_dtype"] == "int16"
    assert config["tf_int_dtype"] == "int16"
    assert config["vocabulary"] == "test/tokenizer/vocabulary"
    assert config["type"] == "Tokenizer"
