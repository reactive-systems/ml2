"""Propositional satisfiability dataset"""

import csv
import logging

from ..datasets import CSVDataset, SplitDataset
from ..globals import PROP_SAT_ALIASES, PROP_SAT_PROJECT_NAME
from ..registry import register_type
from .prop_sat_problem import PropSatProblem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class PropSatDataset(CSVDataset[PropSatProblem]):
    """Propositional satisfiability dataset"""

    ALIASES = PROP_SAT_ALIASES

    def __init__(self, project: str = PROP_SAT_PROJECT_NAME, **kwargs):
        super().__init__(project=project, **kwargs)


@register_type
class PropSatSplitDataset(SplitDataset):
    ALIASES = PROP_SAT_ALIASES

    def __init__(self, project: str = PROP_SAT_PROJECT_NAME, **kwargs):
        super().__init__(project=project, **kwargs)

    # def convert_legacy_format(self) -> None:
    #     for name, split in self._splits.items():
    #         logger.info("Converting %s data", name)
    #         split.df["formula"] = split.df["formula"].map(
    #             lambda s: PropFormula.from_str(
    #                 " ".join(s).replace("x o r", "^").replace("< ", "<").replace(" >", ">"),
    #                 notation="prefix",
    #             ).to_str(notation="infix")
    #         )
    #         split.df["assignment"] = split.df["assignment"].map(
    #             lambda a: ",".join([a[i] + a[i + 1] for i in range(0, len(a), 2)])
    #         )

    # def prefix_to_infix(self) -> None:
    #     for name, split in self._splits.items():
    #         logger.info("Converting %s data to infix", name)
    #         split.df["formula"] = split.df["formula"].map(
    #             lambda s: PropFormula.from_str(
    #                 " ".join(s).replace("< ", "<").replace(" >", ">"), notation="prefix"
    #             ).to_str(notation="infix")
    #         )


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
