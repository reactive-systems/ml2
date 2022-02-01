"""Propositional satisfiability data"""

import csv
import json
import logging
import os
import pandas as pd

from ..data import SupervisedData, SplitSupervisedData
from ..globals import PROP_SAT_ALIASES, PROP_SAT_BUCKET_DIR, PROP_SAT_WANDB_PROJECT
from .prop_formula import PropFormula

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropSatData(SupervisedData):
    """Propositional satisfiability data"""

    ALIASES = PROP_SAT_ALIASES
    BUCKET_DIR = PROP_SAT_BUCKET_DIR
    WANDB_PROJECT = PROP_SAT_WANDB_PROJECT

    def generator(self):
        for _, row in self.data_frame.iterrows():
            yield row["formula"], row["assignment"]

    def save_to_path(self, path: str) -> None:
        self.data_frame.to_csv(path + ".csv", index=False, quoting=csv.QUOTE_ALL)

    @classmethod
    def load_from_path(cls, path: str):
        data_frame = pd.read_csv(path, dtype={"formula": str, "sat": int, "assignment": str})
        return cls(data_frame)


class PropSatSplitData(SplitSupervisedData):

    ALIASES = PROP_SAT_ALIASES
    BUCKET_DIR = PROP_SAT_BUCKET_DIR
    WANDB_PROJECT = PROP_SAT_WANDB_PROJECT

    def convert_legacy_format(self) -> None:
        for name, split in self._splits.items():
            logger.info("Converting %s data", name)
            split.data_frame["formula"] = split.data_frame["formula"].map(
                lambda s: PropFormula.from_str(
                    " ".join(s).replace("x o r", "^").replace("< ", "<").replace(" >", ">"),
                    notation="prefix",
                ).to_str(notation="infix")
            )
            split.data_frame["assignment"] = split.data_frame["assignment"].map(
                lambda a: ",".join([a[i] + a[i + 1] for i in range(0, len(a), 2)])
            )

    def prefix_to_infix(self) -> None:
        for name, split in self._splits.items():
            logger.info("Converting %s data to infix", name)
            split.data_frame["formula"] = split.data_frame["formula"].map(
                lambda s: PropFormula.from_str(
                    " ".join(s).replace("< ", "<").replace(" >", ">"), notation="prefix"
                ).to_str(notation="infix")
            )

    @classmethod
    def load_from_path(cls, path: str, splits: list = None):
        if not splits:
            splits = ["train", "val", "test"]

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
            logger.info("Read in metadata")

        split_dataset = cls(metadata=metadata)
        for split in splits:
            split_path = os.path.join(path, split + ".csv")
            split_dataset[split] = PropSatData.load_from_path(split_path)
        return split_dataset


def legacy_format_to_csv(filepath: str) -> None:
    """Converts a txt file in formula\nassignment\n format into a csv file with comma-separated
    formula, sat, and assignment columns
    """
    with open(filepath, "r") as txt_file, open(filepath[:-3] + "csv", "w") as csv_file:
        csv_file_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        csv_file_writer.writerow(["formula", "sat", "assignment"])
        for line in txt_file:
            if line == "\n":
                return
            formula = line.strip()
            assignment = next(txt_file).strip()
            csv_file_writer.writerow([formula, "1", assignment])
