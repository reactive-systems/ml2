"""LTL satisfiability data"""

import csv
import json
import logging
import os
import pandas as pd
from tqdm import tqdm

from ...data import SupervisedData, SplitSupervisedData
from ...globals import LTL_SAT_ALIASES, LTL_SAT_BUCKET_DIR, LTL_SAT_WANDB_PROJECT
from ...trace import SymbolicTrace
from ..ltl_formula import LTLFormula

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSatData(SupervisedData):
    """LTL satisfiability data"""

    ALIASES = LTL_SAT_ALIASES
    BUCKET_DIR = LTL_SAT_BUCKET_DIR
    WANDB_PROJECT = LTL_SAT_WANDB_PROJECT

    def generator(self):
        for _, row in self.data_frame.iterrows():
            yield row["formula"], row["trace"]

    def save_to_path(self, path: str) -> None:
        self.data_frame.to_csv(path + ".csv", index=False, quoting=csv.QUOTE_ALL)

    @classmethod
    def load_from_path(cls, path: str):
        data_frame = pd.read_csv(path, dtype={"formula": str, "sat": int, "trace": str})
        return cls(data_frame)


class LTLSatSplitData(SplitSupervisedData):

    ALIASES = LTL_SAT_ALIASES
    BUCKET_DIR = LTL_SAT_BUCKET_DIR
    WANDB_PROJECT = LTL_SAT_WANDB_PROJECT

    def mc(self):
        # TODO fix cyclic dependency
        # TODO implement case where formula not satisfiable
        from ...tools.spot import Spot

        spot = Spot()
        counters = {}
        for split in self.split_names:
            with tqdm(desc=split) as pbar:
                for formula, sat, trace in self[split].generator():
                    if sat:
                        trace = SymbolicTrace.from_str(trace).to_str(spot=True)
                        result = spot.mc_trace(formula, trace)
                        counters[result.value] = counters.get(result.value, 0) + 1
                        pbar.update()
                        pbar.set_postfix(counters)
                    else:
                        raise NotImplementedError()

    def space(self):
        for name, split in self._splits.items():
            logger.info("Spacing %s data", name)
            split.data_frame["formula"] = split.data_frame["formula"].map(lambda s: " ".join(s))
            split.data_frame["trace"] = split.data_frame["trace"].map(lambda s: " ".join(s))

    def prefix_to_infix(self):
        for name, split in self._splits.items():
            logger.info("Converting %s data to infix", name)
            split.data_frame["formula"] = split.data_frame["formula"].map(
                lambda s: LTLFormula.from_str(
                    " ".join(s).replace(">", "->"), notation="prefix"
                ).to_str(notation="infix")
            )
            split.data_frame["trace"] = split.data_frame["trace"].map(
                lambda t: SymbolicTrace.from_str(" ".join(t), notation="prefix").to_str(
                    notation="infix"
                )
            )

    def full_parentheses(self):
        for name, split in self._splits.items():
            logger.info("Adding full parentheses to %s data", name)
            split.data_frame["formula"] = split.data_frame["formula"].map()

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
            split_dataset[split] = LTLSatData.load_from_path(split_path)
        return split_dataset


def legacy_format_to_csv(filepath: str) -> None:
    """Converts a txt file in formula\ntrace\n format into a csv file with comma-separated
    formula, sat, and trace columns
    """
    with open(filepath, "r") as txt_file, open(filepath[:-3] + "csv", "w") as csv_file:
        csv_file_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        csv_file_writer.writerow(["formula", "sat", "trace"])
        for line in txt_file:
            if line == "\n":
                return
            formula = line.strip()
            trace = next(txt_file).strip()
            csv_file_writer.writerow([formula, "1", trace])
