"""LTL satisfiability dataset"""

import csv
import logging

import pandas as pd
from tqdm import tqdm

from ...datasets import CSVDataset
from ...registry import register_type
from ...trace import SymbolicTrace
from ..ltl_formula import LTLFormula
from .ltl_sat_status import LTLSatStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class LTLSatDataset(CSVDataset):
    """LTL satisfiability dataset"""

    def __init__(
        self,
        notation: str = None,
        **kwargs,
    ):
        self.notation = notation
        super().__init__(**kwargs)

    def __getitem__(self, key: int):
        if self.notation is not None:
            return self.dtype.from_csv_fields(self.df.iloc[key].to_dict(), notation=self.notation)
        else:
            return self.dtype.from_csv_fields(self.df.iloc[key].to_dict())

    def add_sample(self, sample, **kwargs) -> None:
        if self.notation is not None:
            df_dictionary = pd.DataFrame([sample.to_csv_fields(notation=self.notation, **kwargs)])
        else:
            df_dictionary = pd.DataFrame([sample.to_csv_fields(**kwargs)])
        self.df = pd.concat([self.df, df_dictionary], ignore_index=True)

    def generator(self, **kwargs):
        for _, row in self.df.iterrows():
            row = row.dropna()
            if self.notation is not None:
                yield self.dtype.from_csv_fields(row.to_dict(), notation=self.notation)
            else:
                yield self.dtype.from_csv_fields(row.to_dict())

    def mc(self):
        from ...tools.spot import Spot

        spot = Spot()
        counters = {}
        with tqdm(desc=self.name) as pbar:
            for sample in self.generator():
                if self.status == LTLSatStatus("satisfiable"):
                    result = spot.mc_trace(sample.formula, sample.trace)
                    counters[result.value] = counters.get(result.value, 0) + 1
                    pbar.update()
                    pbar.set_postfix(counters)
                else:
                    raise NotImplementedError()

    def space(self):
        logger.info("Spacing %s data", self.name)
        self.df["formula"] = self.df["formula"].map(lambda s: " ".join(s))
        self.df["trace"] = self.df["trace"].map(lambda s: " ".join(s))

    def prefix_to_infix(self):
        logger.info("Converting %s data to infix", self.name)
        self.df["formula"] = self.df["formula"].map(
            lambda s: LTLFormula.from_str(
                " ".join(s).replace(">", "->"), notation="prefix"
            ).to_str(notation="infix")
        )
        self.df["trace"] = self.df["trace"].map(
            lambda t: SymbolicTrace.from_str(" ".join(t), notation="prefix").to_str(
                notation="infix"
            )
        )


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
