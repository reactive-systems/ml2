"""LTL specification dataset"""

import logging
import os
import random
from copy import copy, deepcopy
from statistics import mean, median
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import pandas as pd
from tqdm import tqdm

from ...datasets import Dataset
from ...globals import LTL_SPEC_ALIASES
from ...registry import register_type
from ..ltl_lexer import lex_ltl
from .decomp_ltl_spec import DecompLTLSpec
from .ltl_spec import LTLSpec
from .ltl_spec_csv_dataset import LTLSpecCSVDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class LTLSpecDataset(Dataset[DecompLTLSpec]):
    ALIASES = LTL_SPEC_ALIASES

    def __init__(
        self,
        dataset: List[DecompLTLSpec],
        name: str,
        dtype: Type[DecompLTLSpec] = DecompLTLSpec,
        **kwargs,
    ):
        self.dataset = dataset

        super().__init__(name=name, dtype=dtype, **kwargs)
        logger.info("Successfully constructed dataset of %d LTL specifications", len(self.dataset))

        # TODO needs data preprocessor
        # self.filter(
        #     ast_size=(0, 25),
        #     num_inputs=(0, 5),
        #     num_outputs=(0, 5),
        #     num_properties=(0, 12),
        #     inplace=True,
        # )
        # self.rename_aps(
        #     input_aps=["i0", "i1", "i2", "i3", "i4"], output_aps=["o0", "o1", "o2", "o3", "o4"]
        # )
        # for spec in self.dataset:
        #     spec.inputs = ["i0", "i1", "i2", "i3", "i4"]
        #     spec.outputs = ["o0", "o1", "o2", "o3", "o4"]

    @property
    def df(self):
        return pd.DataFrame([s.to_csv_fields() for s in self.dataset])

    @property
    def size(self) -> int:
        return len(self.dataset)

    def add_sample(self, sample: DecompLTLSpec, **kwargs) -> None:
        self.dataset.append(sample)

    def config_postprocessors(self) -> list:
        def postprocess_data(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("dataset", None)
            annotations.pop("dataset", None)

        return [postprocess_data] + super().config_postprocessors()

    def filter(
        self,
        ast_size: Tuple[Optional[int], Optional[int]] = (None, None),
        num_inputs: Tuple[Optional[int], Optional[int]] = (None, None),
        num_outputs: Tuple[Optional[int], Optional[int]] = (None, None),
        num_properties: Tuple[Optional[int], Optional[int]] = (None, None),
        inplace: bool = False,
    ) -> Optional["LTLSpecDataset"]:
        counter = {
            "max_ast_size": 0,
            "max_inputs": 0,
            "max_outputs": 0,
            "min_ast_size": 0,
            "min_inputs": 0,
            "min_outputs": 0,
            "min_properties": 0,
            "max_properties": 0,
        }
        t = tqdm(desc="Filtering", total=len(self.dataset), postfix=counter)

        def filter_specs(spec):
            filtered_out: bool = False
            try:
                if not (num_inputs[0] is None or spec.num_inputs >= num_inputs[0]):
                    counter["min_inputs"] += 1
                    filtered_out = True
                if not (num_inputs[1] is None or spec.num_inputs <= num_inputs[1]):
                    counter["max_inputs"] += 1
                    filtered_out = True
                if not (num_outputs[0] is None or spec.num_outputs >= num_outputs[0]):
                    counter["min_outputs"] += 1
                    filtered_out = True
                if not (num_outputs[1] is None or spec.num_outputs <= num_outputs[1]):
                    counter["max_outputs"] += 1
                    filtered_out = True
                if not (
                    num_properties[0] is None
                    or (len(spec.guarantees) + len(spec.assumptions)) >= num_properties[0]
                ):
                    counter["min_properties"] += 1
                    filtered_out = True
                if not (
                    num_properties[1] is None
                    or (len(spec.guarantees) + len(spec.assumptions)) <= num_properties[1]
                ):
                    counter["max_properties"] += 1
                    filtered_out = True
                if not (
                    ast_size[0] is None
                    or min([s.ast.size() for s in (spec.guarantees + spec.assumptions)])
                    >= ast_size[0]
                ):
                    counter["min_ast_size"] += 1
                    filtered_out = True
                if not (
                    ast_size[1] is None
                    or max([s.ast.size() for s in (spec.guarantees + spec.assumptions)])
                    <= ast_size[1]
                ):
                    counter["max_ast_size"] += 1
                    filtered_out = True
            except RecursionError:
                logger.info("Recursion Error")
                counter["max_ast_size"] += 1
                filtered_out = True
            t.set_postfix(counter, refresh=False)
            t.update()
            return not filtered_out

        dataset = list(filter(filter_specs, self.dataset))

        logger.info(
            "%d AST too large, %d AST too small, %d too many inputs,  %d not enough inputs, %d too many outputs, %d not enough outputs, %d too many properties, %d not enough properties. %d specs remaining (%.2f).",
            counter["max_ast_size"],
            counter["min_ast_size"],
            counter["max_inputs"],
            counter["min_inputs"],
            counter["max_outputs"],
            counter["min_outputs"],
            counter["max_properties"],
            counter["min_properties"],
            len(dataset),
            len(dataset) / len(self.dataset),
        )

        if not inplace:
            new_dataset: LTLSpecDataset = copy(self)
            new_dataset.dataset = dataset
            return new_dataset
        else:
            self.dataset = dataset

    def generator(self, **kwargs) -> Generator[DecompLTLSpec, None, None]:
        for sample in self.dataset:
            yield sample

    def sample(self, n: int) -> None:
        self.dataset = random.sample(self.dataset, n)

    def rename_aps(
        self,
        input_aps: List[str],
        output_aps: List[str],
        random: bool = True,
        renaming: Dict[str, str] = None,
    ) -> None:
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

    def assumptions(self) -> LTLSpecCSVDataset:
        assumptions = LTLSpecCSVDataset(name="assumptions", dtype=LTLSpec, project=self.project)

        for s in self.dataset:
            uniqueness = []
            for a in s.assumptions.sub_exprs:
                try:
                    a_c: LTLSpec = deepcopy(a)
                    a_c.reset_aps()
                    a_c.rename_aps(random=False)
                    if uniqueness.count(a_c.unique_id()) > 3:
                        continue
                    uniqueness.append(a_c.unique_id())
                    assumptions.add_sample(a_c)
                except RecursionError as e:
                    print(
                        e,
                        "for specification",
                        s.name,
                        ", assumption",
                        a.to_str(),
                        ". Sample ignored.",
                    )
        return assumptions

    def guarantees(self) -> LTLSpecCSVDataset:
        guarantees = LTLSpecCSVDataset(name="guarantees", dtype=LTLSpec, project=self.project)

        for s in self.dataset:
            uniqueness = []
            for g in s.guarantees.sub_exprs:
                try:
                    g_c: LTLSpec = deepcopy(g)
                    g_c.reset_aps()
                    g_c.rename_aps(random=False)
                    if uniqueness.count(g_c.unique_id()) > 3:
                        continue
                    guarantees.add_sample(g_c)
                    uniqueness.append(g_c.unique_id())
                except RecursionError as e:
                    print(
                        e,
                        "for specification",
                        s.name,
                        ", guarantee",
                        g.to_str(),
                        ". Sample ignored.",
                    )
        return guarantees

    def properties(self) -> Dict[str, LTLSpecCSVDataset]:
        return {"assumptions": self.assumptions(), "guarantees": self.guarantees()}

    def assumptions_old(self, unique: bool = False):
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

    def guarantees_old(self, unique: bool = False):
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

    def save_to_path(self, path: str, file_format: str = "tlsf", **kwargs) -> None:
        for spec in self.dataset:
            spec.to_file(file_dir=path, filename=None, file_format=file_format)

    @classmethod
    def config_preprocessors(
        cls,
    ) -> list:
        def preprocess_bosy_files(config: Dict[str, Any], annotations: Dict[str, type]):
            name_as_id = config.pop("name_as_id", False)
            path = cls.local_path_from_name(name=config["name"], project=config["project"])
            dataset = cls.list_from_bosy_files(path=path, name_as_id=name_as_id)
            config["dataset"] = dataset

        return super().config_preprocessors() + [preprocess_bosy_files]

    @classmethod
    def from_bosy_files(
        cls, path: str, ltl_spec_filter=None, name_as_id: bool = False, **kwargs
    ) -> "LTLSpecDataset":
        """Constructs a dataset of LTL specifications from a directory with BoSy input files"""
        dataset = cls.list_from_bosy_files(
            path=path, ltl_spec_filter=ltl_spec_filter, name_as_id=name_as_id
        )
        return cls(dataset=dataset, **kwargs)

    @classmethod
    def from_tlsf_files(
        cls,
        path: str,
        ltl_spec_filter=None,
        port: int = 50051,
        start_container: bool = True,
        name_as_id: bool = False,
        **kwargs,
    ) -> "LTLSpecDataset":
        """Constructs a dataset of LTL specifications from a directory with BoSy input files"""

        dataset = cls.list_from_tlsf_files(
            path=path,
            ltl_spec_filter=ltl_spec_filter,
            port=port,
            start_container=start_container,
            name_as_id=name_as_id,
        )
        return cls(dataset=dataset, **kwargs)

    @staticmethod
    def list_from_bosy_files(
        path: str, ltl_spec_filter=None, name_as_id: bool = False
    ) -> List[DecompLTLSpec]:
        """Constructs a list of LTL specifications from a directory with BoSy input files"""
        dataset = []
        for file in os.listdir(path):
            if file.endswith(".json") and file != "config.json" and file != "metadata.json":
                ltl_spec = DecompLTLSpec.from_bosy_file(os.path.join(path, file))
                if name_as_id:
                    ltl_spec._unique_id_value = ltl_spec.name
                if not ltl_spec_filter or ltl_spec_filter(ltl_spec):
                    dataset.append(ltl_spec)
        return dataset

    @staticmethod
    def list_from_tlsf_files(
        path: str,
        ltl_spec_filter=None,
        port: int = 50051,
        start_containerized_service: bool = True,
        name_as_id: bool = False,
    ) -> List[DecompLTLSpec]:
        """Constructs a list of LTL specifications from a directory with BoSy input files"""

        from ml2.tools.syfco.syfco import Syfco

        dataset = []
        syfco = Syfco(port=port, start_containerized_service=start_containerized_service)
        for file in os.listdir(path):
            if file.endswith(".tlsf"):
                ltl_spec = syfco.from_tlsf_file(os.path.join(path, file))
                if name_as_id:
                    ltl_spec._unique_id_value = ltl_spec.name
                if not ltl_spec_filter or ltl_spec_filter(ltl_spec):
                    dataset.append(ltl_spec)
        return dataset
