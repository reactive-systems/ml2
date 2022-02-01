"""LTL specification data"""

import logging
from statistics import mean
from statistics import median
import os

from ...data import Data
from ...globals import LTL_SPEC_ALIASES, LTL_SPEC_BUCKET_DIR, LTL_SPEC_WANDB_PROJECT
from ..ltl_lexer import lex_ltl
from .ltl_spec import LTLSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSpecData(Data):

    ALIASES = LTL_SPEC_ALIASES
    BUCKET_DIR = LTL_SPEC_BUCKET_DIR
    WANDB_PROJECT = LTL_SPEC_WANDB_PROJECT

    def __init__(self, dataset, metadata: dict = None):
        self.dataset = dataset
        self.metadata = metadata
        logger.info("Successfully constructed dataset of %d LTL specifications", len(self.dataset))

    def rename_aps(self, input_aps, output_aps, random=True, renaming=None):
        for specification in self.dataset:
            specification.rename_aps(input_aps, output_aps, random, renaming)
        logger.info(
            (
                "Renamed input atomic propositions to %s and renamed "
                "output atomic propositions to %s"
            ),
            input_aps,
            output_aps,
        )

    def stats(self):
        """Computes statistics of the dataset"""
        num_inputs = [spec.num_inputs for spec in self.dataset]
        num_outputs = [spec.num_outputs for spec in self.dataset]
        num_assumptions = [spec.num_assumptions for spec in self.dataset]
        num_guarantees = [spec.num_guarantees for spec in self.dataset]
        print(f"Computed statistics of {len(self.dataset)} specifications")
        for key, values in [
            ("inputs", num_inputs),
            ("outputs", num_outputs),
            ("assumptions", num_assumptions),
            ("guarantees", num_guarantees),
        ]:
            print(f"Number of {key}")
            print(
                (
                    f"minimum: {min(values)} maximum: {max(values)} "
                    f"mean: {mean(values)} median: {median(values)} "
                    f"total: {sum(values)}"
                )
            )

    def assumptions(self, unique=False):
        """Returns a list of all (syntactically unique) assumptions including
        inputs and outputs that appear in the dataset.
        """
        result = []
        assumptions = set()
        for spec in self.dataset:
            for assumption in spec.assumptions:
                if unique:
                    if assumption in assumptions:
                        continue
                    assumptions.add(assumption)
                # TODO move functionalty into specification class
                tokens = lex_ltl(assumption)
                result.append(
                    {
                        "inputs": [i for i in spec.inputs if i in tokens],
                        "outputs": [o for o in spec.outputs if o in tokens],
                        "assumption": assumption,
                    }
                )
        logger.info("Bundled %d %s assumptions", len(result), "unique" if unique else "non-unique")
        return result

    def guarantees(self, unique=False):
        """Returns a list of all (syntactically unique) guarantees including
        inputs and outputs that appear in the dataset
        """
        result = []
        guarantees = set()
        for spec in self.dataset:
            for guarantee in spec.guarantees:
                if unique:
                    if guarantee in guarantees:
                        continue
                    guarantees.add(guarantee)
                # TODO move functionalty into specification class
                tokens = lex_ltl(guarantee)
                result.append(
                    {
                        "inputs": [i for i in spec.inputs if i in tokens],
                        "outputs": [o for o in spec.outputs if o in tokens],
                        "guarantee": guarantee,
                    }
                )
        logger.info("Bundled %d %s guarantees", len(result), "unique" if unique else "non-unique")
        return result

    def save_to_path(self, path: str) -> None:
        for spec in self.dataset:
            spec.to_file(path)
        logger.info("Saved %d files to %s", len(self.dataset), path)

    @classmethod
    def from_bosy_files(cls, directory: str, ltl_spec_filter=None):
        """Constructs a dataset of LTL specifications from a directory with BoSy input files"""
        dataset = []
        for file in os.listdir(directory):
            if file.endswith(".json"):
                ltl_spec = LTLSpec.from_bosy_file(os.path.join(directory, file))
                if not ltl_spec_filter or ltl_spec_filter(ltl_spec):
                    dataset.append(ltl_spec)
        return cls(dataset)

    @classmethod
    def load_from_path(cls, path: str):
        return cls.from_bosy_files(path)
