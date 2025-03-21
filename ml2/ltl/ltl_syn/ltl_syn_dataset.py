"""LTL synthesis dataset"""

import copy
import csv
import logging
import os
import re
from typing import Any, Dict, Generator, Self

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from ... import aiger
from ...artifact import Artifact
from ...datasets import SplitDataset
from ...datasets.csv_dataset import CSVDataset
from ...datasets.stats import stats_from_counts
from ...datasets.utils import to_csv_str
from ...globals import LTL_SYN_ALIASES, LTL_SYN_PROJECT_NAME
from ...registry import register_type
from ..ltl_spec import DecompLTLSpec, LTLSpec
from .decomp_ltl_syn_problem import DecompLTLSynProblem, LTLSynSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# circuit statistics keys
MAX_VAR_INDEX = "MAX VARIABLE INDEX"
NUM_INPUTS = "NUM INPUTS"
NUM_LATCHES = "NUM LATCHES"
NUM_OUTPUTS = "NUM OUTPUTS"
NUM_AND_GATES = "NUM AND GATES"


def curriculum_sample_to_lists(sample: dict):
    # properties, propertie types, inputs, outputs, realizable values, circuits
    l = [[], [], [], [], [], []]
    assumption = False
    if sample.get("unrealizable_parent", None):
        l = curriculum_sample_to_lists(sample["unrealizable_parent"])
        assumption = True
        assert len(sample["assumptions"]) > len(sample["unrealizable_parent"]["assumptions"])
    elif sample.get("parent", None):
        l = curriculum_sample_to_lists(sample["parent"])
        assert len(sample["assumptions"]) == len(sample["parent"]["assumptions"])
    if assumption:
        l[0].append(sample["assumptions"][-1])
        l[1].append("A")
    else:
        l[0].append(sample["guarantees"][-1])
        l[1].append("G")
    l[2] = sample["inputs"]
    l[3] = sample["outputs"]
    if "realizable" in sample:
        l[4].append(str(sample["realizable"]))
    else:
        print(sample)
        raise Exception
    l[5].append(to_csv_str(sample["circuit"]))
    return l


def curriculum_sample_to_csv_row(sample: dict):
    l = curriculum_sample_to_lists(sample)
    return [",".join(c) for c in l]


@register_type
class LTLSynDataset(CSVDataset[DecompLTLSynProblem]):
    ALIASES = LTL_SYN_ALIASES

    def __init__(self, project: str = LTL_SYN_PROJECT_NAME, **kwargs):
        super().__init__(project=project, **kwargs)

    def stats(self, **kwargs):
        def spec_stats(spec: DecompLTLSpec) -> Dict[str, float]:
            flattened: Dict[str, float] = {}
            for k, v in spec.property_stats().items():
                flattened = {**flattened, **{k + "_" + ki: vi for ki, vi in v.items()}}
            return {
                **flattened,
                "spec_num_inputs": spec.num_inputs,
                "spec_num_outputs": spec.num_outputs,
                "spec_num_aps": spec.num_aps,
                "spec_num_guarantees": len(spec.guarantees),
                "spec_num_assumptions": len(spec.assumptions),
                "spec_num_properties": len(spec.guarantees) + len(spec.assumptions),
                "spec_size": spec.size(),
            }

        def circuit_stats(circ: aiger.AIGERCircuit) -> Dict[str, float]:
            return {
                "circ_num_inputs": circ.num_inputs,
                "circ_num_outputs": circ.num_outputs,
                "circ_num_latches": circ.num_latches,
                "circ_num_ands": circ.num_ands,
                "circ_max_var_id": circ.max_var_id,
            }

        def solution_stats(solution: LTLSynSolution) -> Dict[str, float]:
            return {**circuit_stats(solution.circuit), "realizable": solution.status.to_int()}

        def problem_stats(problem: DecompLTLSynProblem) -> Dict[str, float]:
            return {**spec_stats(problem.ltl_spec), **solution_stats(problem.ltl_syn_solution)}

        def stats_gen() -> Generator[Dict[str, float], None, None]:
            fn = problem_stats if self.dtype == DecompLTLSynProblem else spec_stats
            for el in tqdm(self.generator(), total=self.size):
                yield fn(el)

        stats_list = list(stats_gen())
        return stats_from_counts({k: [dic[k] for dic in stats_list] for k in stats_list[0]})


class LTLSynCurriculumDataset(CSVDataset):
    ALIASES = LTL_SYN_ALIASES

    def __init__(self, project: str = LTL_SYN_PROJECT_NAME, **kwargs):
        super().__init__(project=project, **kwargs)

    def sample_generator(self):
        for _, row in self.df.iterrows():
            props = row["properties"].split(",")
            types = row["type"].split(",")
            reals = row["realizable"].split(",")
            circuits = row["circuits"].split(",")
            sample = {
                "assumptions": [],
                "guarantees": [],
                "inputs": row["inputs"].split(",") if row["inputs"] else [],
                "outputs": row["outputs"].split(",") if row["outputs"] else [],
                "realizable": None,
                "circuit": None,
            }
            for p, t, r, c in zip(props, types, reals, circuits):
                if t == "A":
                    sample["assumptions"].append(p)
                elif t == "G":
                    sample["guarantees"].append(p)
                else:
                    raise ValueError()
                sample["realizable"] = int(r)
                sample["circuit"] = c
                try:
                    if aiger.parse(c).max_var_id > 30:
                        continue
                except Exception:
                    continue
                yield copy.deepcopy(sample)

    def generator(self):
        for sample in self.sample_generator():
            yield LTLSpec.from_dict(sample), sample["circuit"]

    def save_to_path(self, path: str) -> None:
        path = path + ".csv"
        circuit_series = self.df["circuits"]
        self.df["circuits"] = circuit_series.str.replace("\n", "\\n")
        self.df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        self.df["circuits"] = circuit_series

    @classmethod
    def load_from_path(cls, path: str):
        data_frame = pd.read_csv(
            path,
            converters={"circuits": lambda c: str(c).replace("\\n", "\n")},
            dtype={
                "properties": str,
                "type": str,
                "inputs": str,
                "outputs": str,
                "realizable": str,
            },
            keep_default_na=False,
        )
        return cls(data_frame)


@register_type
class LTLSynSplitDataset(SplitDataset):
    ALIASES = LTL_SYN_ALIASES

    def __init__(self, project: str = LTL_SYN_PROJECT_NAME, **kwargs):
        super().__init__(project=project, **kwargs)

    @classmethod
    def load_from_deprecated(cls, name: str, project: str, unsupervised_splits: list[str]) -> Self:
        """
        Load a dataset from the deprecated format.

        Args:
            name (str): The name of the dataset.
            project (str): The project associated with the dataset.
            unsupervised_splits (list[str]): A list of split names that are unsupervised.

        Returns:
            Self: An instance of the class with the loaded dataset.
        """
        path = Artifact.local_path_from_name(name, project=project)

        splits = {}
        for split in [
            ".".join(x.split(".")[:-1])
            for x in os.listdir(path)
            if not os.path.isdir(os.path.join(path, x)) and x.endswith("csv")
        ]:
            split_ds = LTLSynDataset(
                project=project,
                name=os.path.join(name, split),
                dtype=DecompLTLSynProblem if split in unsupervised_splits else DecompLTLSpec,
                filename=split + ".csv",
            )
            file = os.path.join(path, split) + ".csv"
            split_ds.df = pd.read_csv(file, sep=",").fillna("")
            splits[split] = split_ds
        ds = cls(name=name, project=project, dtype=DecompLTLSynProblem, splits=splits)
        return ds

    @classmethod
    def convert_deprecated(cls, name: str, project: str, unsupervised_splits: list[str]) -> Self:
        """
        Converts the deprecated dataset format to the current format.

        This method loads a dataset using a deprecated format, saves it in the current format,
        and removes the old CSV files.

        Args:
            name (str): The name of the dataset.
            project (str): The project to which the dataset belongs.
            unsupervised_splits (list[str]): Indicates the splits that are unsupervised.

        Returns:
            Dataset: The converted dataset object.
        """
        path = Artifact.local_path_from_name(name, project=project)

        ds = cls.load_from_deprecated(name, project, unsupervised_splits)
        ds.save(recurse=True)
        for split in [
            ".".join(x.split(".")[:-1])
            for x in os.listdir(path)
            if not os.path.isdir(os.path.join(path, x)) and x.endswith("csv")
        ]:
            os.remove(os.path.join(path, split) + ".csv")
        return ds


def space_dataset(path: str, num_next: int = 2):
    split_dataset = LTLSynSplitDataset.load_from_path(path)
    fin_regex = re.compile("(F)([A-Za-z])")
    glob_regex = re.compile("(G)([A-Za-z])")
    next_regex = re.compile("(X)([A-Za-z])")
    for split in ["train", "val", "test", "timeouts"]:
        df = split_dataset[split].data_frame
        df["guarantees"] = df["guarantees"].str.replace(fin_regex, r"\g<1> \g<2>")
        df["guarantees"] = df["guarantees"].str.replace(glob_regex, r"\g<1> \g<2>")
        for _ in range(num_next):
            df["guarantees"] = df["guarantees"].str.replace(next_regex, r"\g<1> \g<2>")
    split_dataset.save_to_path(path)


def model_check_ltl_syn_dataset(name: str, timeout: float = 10.0) -> None:
    split_dataset = LTLSynSplitDataset.load(name)
    from ...tools.nuxmv import nuXmv

    nuxmv = nuXmv(port=50051)
    for split in ["train", "val", "test"]:
        counters = {}
        with tqdm(desc=split) as pbar:
            for sample in split_dataset[split].generator():
                result = nuxmv.model_check(
                    sample.ltl_spec,
                    sample.ltl_syn_solution.circuit,
                    realizable=sample.ltl_syn_solution.status.realizable,
                    timeout=timeout,
                )
                counters[result.value] = counters.get(result.value, 0) + 1
                if result.value == "error":
                    print(sample)
                    return
                pbar.update()
                pbar.set_postfix(counters)
