"""Load vocabulary test"""

import pytest

from ml2.tokenizers import Vocabulary


@pytest.mark.gcp
def test_load_vocab():
    vocab = Vocabulary.load(
        name="ht-12/train/pipe/input-tokenizer/sub-tokenizer/vocab", project="ltl-syn"
    )
    assert vocab.tokens_to_ids(["i0", "U", "o0"]) == [11, 8, 26]


@pytest.mark.gcp
def test_load_vocab_with_kwargs():
    NEW_TOKEN_TO_ID = {"a": 0, "b": 1, "U": 2}
    vocab = Vocabulary.load(
        name="ht-12/train/pipe/input-tokenizer/sub-tokenizer/vocab",
        project="ltl-syn",
        token_to_id=NEW_TOKEN_TO_ID,
    )
    assert vocab.tokens_to_ids(["a", "U", "b"]) == [0, 2, 1]
