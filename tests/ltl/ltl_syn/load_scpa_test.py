"""scpa-2 dataset test"""

import pytest

from ml2.datasets import load_dataset


@pytest.mark.gcp
def test_load_scpa_2():
    ds = load_dataset("ltl-syn/scpa-2")
    assert ds.name == "scpa-2"
    assert "train" in ds
    assert ds["train"].size == 200000
    assert "val" in ds
    assert ds["val"].size == 25000
    assert "test" in ds
    assert ds["test"].size == 25000
    assert "timeouts" in ds
    assert ds["timeouts"].size == 13403


@pytest.mark.gcp
def test_load_with_sample_scpa_2():
    ds = load_dataset("ltl-syn/scpa-2/val", sample=1000)
    assert ds.size == 1000


@pytest.mark.gcp
def test_load_with_shuffle():
    ds_1 = load_dataset("ltl-syn/scpa-2/val")
    ds_2 = load_dataset("ltl-syn/scpa-2/val")
    ds_3 = load_dataset("ltl-syn/scpa-2/val", shuffle=True)
    assert ds_1[0] == ds_2[0]
    # TODO small chance this fails
    assert ds_1[0] != ds_3[0]
