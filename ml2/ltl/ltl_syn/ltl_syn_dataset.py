"""LTL synthesis dataset"""

import copy
import csv
import logging
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from ... import aiger
from ...datasets import SplitDataset
from ...datasets.csv_dataset import CSVDataset
from ...datasets.stats import stats_from_counts
from ...datasets.utils import to_csv_str
from ...globals import LTL_SYN_ALIASES, LTL_SYN_PROJECT_NAME
from ...registry import register_type
from ..ltl_spec import LTLSpec
from .decomp_ltl_syn_problem import DecompLTLSynProblem

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

    def stats(self, splits: list = None):
        counts = {
            MAX_VAR_INDEX: [],
            NUM_INPUTS: [],
            NUM_LATCHES: [],
            NUM_OUTPUTS: [],
            NUM_AND_GATES: [],
        }
        for _, circuit in self.generator(splits):
            (
                num_var_index,
                num_inputs,
                num_latches,
                num_outputs,
                num_and_gates,
            ) = aiger.header_ints_from_str(circuit)
            counts[MAX_VAR_INDEX].append(num_var_index)
            counts[NUM_INPUTS].append(num_inputs)
            counts[NUM_LATCHES].append(num_latches)
            counts[NUM_OUTPUTS].append(num_outputs)
            counts[NUM_AND_GATES].append(num_and_gates)
        return stats_from_counts(counts)

    def plot_stats(self, splits: list = None):
        stats = self.stats(splits)

        def plot_stats(stats: dict, filepath: str, title: str = None):
            file_dir = os.path.dirname(filepath)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
            fig, ax = plt.subplots()
            max_value = stats["max"]
            min_value = stats["min"]
            bins = stats["bins"]
            ax.bar(range(max_value + 1), bins, color="#3071ff", width=0.7, align="center")
            if title:
                ax.set_title(title)
            ax.set_xlim(min_value - 1, max_value + 1)
            ax.set_ylim(0, max(bins) + 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig(filepath, dpi=fig.dpi, facecolor="white", format="eps")
            if title:
                logging.info("%s statistics plotted to %s", title, filepath)

        filepath = os.path.join(self.stats_path, "max_var_id.eps")
        plot_stats(stats[MAX_VAR_INDEX], filepath, "Maximal Variable Index")
        filepath = os.path.join(self.stats_path, "num_inputs.eps")
        plot_stats(stats[NUM_INPUTS], filepath, "Number of Inputs")
        filepath = os.path.join(self.stats_path, "max_num_latches.eps")
        plot_stats(stats[NUM_LATCHES], filepath, "Number of Latches")
        filepath = os.path.join(self.stats_path, "num_outputs.eps")
        plot_stats(stats[NUM_OUTPUTS], filepath, "Number of Outputs")
        filepath = os.path.join(self.stats_path, "num_and_gates.eps")
        plot_stats(stats[NUM_AND_GATES], filepath, "Number of AND Gates")


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
