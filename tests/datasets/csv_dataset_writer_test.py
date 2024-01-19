"""CSV dataset writer test"""

import shutil

import pytest

from ml2.datasets import CSVDatasetWriter, SplitDatasetWriter, load_dataset
from ml2.ltl import LTLFormula


@pytest.mark.gcp
def test_csv_dataset_writer():
    f1 = LTLFormula.from_str("a U b")
    f2 = LTLFormula.from_str("F c")
    f3 = LTLFormula.from_str("G d")
    f4 = LTLFormula.from_str("X X e")

    train_writer = CSVDatasetWriter(
        name="t-1/train",
        dtype=LTLFormula,
        header=["formula", "id_LTLFormula"],
        filename="train.csv",
        project="test",
    )
    val_writer = CSVDatasetWriter(
        name="t-1/val",
        dtype=LTLFormula,
        header=["formula", "id_LTLFormula"],
        filename="val.csv",
        project="test",
    )
    test_writer = CSVDatasetWriter(
        name="t-1/test",
        dtype=LTLFormula,
        header=["formula", "id_LTLFormula"],
        filename="test.csv",
        project="test",
    )

    splits = {
        "train": train_writer,
        "val": val_writer,
        "test": test_writer,
    }
    target_sizes = {"train": 2, "val": 1, "test": 1}

    split_writer = SplitDatasetWriter(
        name="t-1", splits=splits, target_sizes=target_sizes, project="test"
    )

    split_writer.add_sample(f1, notation="infix")
    assert sum(split_writer.split_probs.values()) == 1
    split_writer.add_sample(f2, notation="infix")
    assert sum(split_writer.split_probs.values()) == 1
    split_writer.add_sample(f3, notation="infix")
    assert sum(split_writer.split_probs.values()) == 1
    split_writer.add_sample(f4, notation="infix")
    assert sum(split_writer.split_probs.values()) == 0

    split_writer.close()
    split_writer.save(recurse=True)

    dataset = load_dataset("test/t-1")
    assert dataset.size == 4
    assert dataset["train"].size == 2
    assert dataset["val"].size == 1
    assert dataset["test"].size == 1

    shutil.rmtree(dataset.local_path)
