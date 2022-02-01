"""LTL synthesis data"""

import copy
import csv
import json
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import re
from tqdm import tqdm

import pandas as pd

from ... import aiger
from ...data import SupervisedData, SplitSupervisedData
from ...data.utils import from_csv_str, to_csv_str
from ...data.stats import stats_from_counts
from ...globals import LTL_SYN_ALIASES, LTL_SYN_BUCKET_DIR, LTL_SYN_WANDB_PROJECT
from ...tools.nuxmv import nuXmv
from ..ltl_spec import LTLSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# circuit statistics keys
MAX_VAR_INDEX = "MAX VARIABLE INDEX"
NUM_INPUTS = "NUM INPUTS"
NUM_LATCHES = "NUM LATCHES"
NUM_OUTPUTS = "NUM OUTPUTS"
NUM_AND_GATES = "NUM AND GATES"


def sample_to_csv_row(sample: dict):
    assumptions = ",".join(sample["assumptions"])
    guarantees = ",".join(sample["guarantees"])
    inputs = ",".join(sample["inputs"])
    outputs = ",".join(sample["outputs"])
    realizable = sample["realizable"]
    circuit = to_csv_str(sample["circuit"])
    return [assumptions, guarantees, inputs, outputs, realizable, circuit]


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


def csv_row_to_sample(row: list):
    return {
        "assumptions": row[0].split(","),
        "guarantees": row[1].split(","),
        "inputs": row[2].split(","),
        "outputs": row[3].split(","),
        "realizable": int(row[4]),
        "circuit": from_csv_str(row[5]),
    }


class LTLSynData(SupervisedData):

    ALIASES = LTL_SYN_ALIASES
    BUCKET_DIR = LTL_SYN_BUCKET_DIR
    WANDB_PROJECT = LTL_SYN_WANDB_PROJECT

    def sample_generator(self):
        for _, row in self.data_frame.iterrows():
            sample = {
                "assumptions": row["assumptions"].split(",")
                if "assumptions" in row and row["assumptions"]
                else [],  # key in dict check for data that does not contain assumptions
                "guarantees": row["guarantees"].split(",") if row["guarantees"] else [],
                "inputs": row["inputs"].split(",") if row["inputs"] else [],
                "outputs": row["outputs"].split(",") if row["outputs"] else [],
                "realizable": row["realizable"],
                "circuit": row["circuit"],
            }
            yield sample

    def generator(self):
        for sample in self.sample_generator():
            yield LTLSpec.from_dict(sample), sample["circuit"]

    def save_to_path(self, path: str) -> None:
        path = path + ".csv"
        circuit_series = self.data_frame["circuit"]
        self.data_frame["circuit"] = circuit_series.str.replace("\n", "\\n")
        self.data_frame.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        self.data_frame["circuit"] = circuit_series

    @classmethod
    def load_from_path(cls, path: str):
        data_frame = pd.read_csv(
            path,
            converters={"circuit": lambda c: str(c).replace("\\n", "\n")},
            dtype={
                "assumptions": str,
                "guarantees": str,
                "inputs": str,
                "outputs": str,
                "realizable": int,
            },
            keep_default_na=False,
        )
        return cls(data_frame)


class LTLSynCurriculumData(SupervisedData):

    ALIASES = LTL_SYN_ALIASES
    BUCKET_DIR = LTL_SYN_BUCKET_DIR
    WANDB_PROJECT = LTL_SYN_WANDB_PROJECT

    def sample_generator(self):
        for _, row in self.data_frame.iterrows():
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
        circuit_series = self.data_frame["circuits"]
        self.data_frame["circuits"] = circuit_series.str.replace("\n", "\\n")
        self.data_frame.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        self.data_frame["circuits"] = circuit_series

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


class LTLSynSplitData(SplitSupervisedData):

    ALIASES = LTL_SYN_ALIASES
    BUCKET_DIR = LTL_SYN_BUCKET_DIR
    WANDB_PROJECT = LTL_SYN_WANDB_PROJECT

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

        filepath = os.path.join(self.stats_path(self.name), "max_var_id.eps")
        plot_stats(stats[MAX_VAR_INDEX], filepath, "Maximal Variable Index")
        filepath = os.path.join(self.stats_path(self.name), "num_inputs.eps")
        plot_stats(stats[NUM_INPUTS], filepath, "Number of Inputs")
        filepath = os.path.join(self.stats_path(self.name), "max_num_latches.eps")
        plot_stats(stats[NUM_LATCHES], filepath, "Number of Latches")
        filepath = os.path.join(self.stats_path(self.name), "num_outputs.eps")
        plot_stats(stats[NUM_OUTPUTS], filepath, "Number of Outputs")
        filepath = os.path.join(self.stats_path(self.name), "num_and_gates.eps")
        plot_stats(stats[NUM_AND_GATES], filepath, "Number of AND Gates")

    @classmethod
    def load_from_path(cls, path: str, splits: list = None):
        if not splits:
            splits = ["train", "val", "test", "timeouts"]

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
            logger.info("Read in metadata")

        split_dataset = cls(metadata=metadata)
        for split in splits:
            split_path = os.path.join(path, split + ".csv")
            if metadata.get("curriculum", False):
                split_dataset[split] = LTLSynCurriculumData.load_from_path(split_path)
            else:
                split_dataset[split] = LTLSynData.load_from_path(split_path)
        return split_dataset


def space_dataset(path: str, num_next: int = 2):
    # TODO test function
    split_dataset = LTLSynSplitData.load_from_path(path)
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


def model_check_data(name: str, timeout: float = 10.0):
    split_dataset = LTLSynSplitData.load(name)
    nuxmv = nuXmv(port=50051)
    for split in ["train", "val", "test"]:
        counters = {}
        with tqdm(desc=split) as pbar:
            for sample in split_dataset[split].sample_generator():
                result = nuxmv.model_check(
                    LTLSpec.from_dict(sample),
                    sample["circuit"] + "\n",
                    realizable=sample["realizable"],
                    timeout=timeout,
                )
                counters[result.value] = counters.get(result.value, 0) + 1
                if result.value == "error":
                    print(sample)
                    return
                pbar.update()
                pbar.set_postfix(counters)
