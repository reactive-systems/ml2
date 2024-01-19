"""Common functionality for synthesis data generation"""

import argparse
import csv
import logging
import os

import numpy as np
import ray

from ... import aiger
from ...datasets.utils import to_csv_str
from ..ltl_spec import DecompLTLSpec
from .ltl_syn_dataset import curriculum_sample_to_csv_row
from .ltl_syn_problem import LTLSynProblem, LTLSynSolution

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def sample_to_csv_row(sample: dict):
    assumptions = ",".join(sample["assumptions"])
    guarantees = ",".join(sample["guarantees"])
    inputs = ",".join(sample["inputs"])
    outputs = ",".join(sample["outputs"])
    realizable = sample["realizable"]
    circuit = to_csv_str(sample["circuit"])
    return [assumptions, guarantees, inputs, outputs, realizable, circuit]


def sample_to_csv_fields(sample: dict):
    spec = DecompLTLSpec.from_dict(sample)
    sol: LTLSynSolution = LTLSynSolution.from_csv_fields(sample)  # type: ignore
    prob: LTLSynProblem = LTLSynProblem(spec, sol)
    return prob.to_csv_fields(notation="infix")


def add_ltl_syn_data_gen_args(parser):
    parser.add_argument(
        "--all-aps",
        action="store_true",
        help=("circuits incorporate all AP contained in the set" "of guarantees"),
    )
    parser.add_argument(
        "-c", "--curriculum", action="store_true", help="generate data for curriculum learning"
    )
    parser.add_argument(
        "--max-property-size",
        type=int,
        default=25,
        help=("max size for each properties"),
    )
    parser.add_argument(
        "--max-frac-ands",
        type=float,
        default=None,
        metavar="max",
        help=("maximal fraction of circuits with" "0,1,2,... ANG gates"),
    )
    parser.add_argument(
        "--num-ands",
        action=argparse_min_max(),
        nargs="*",
        default=(0, None),
        metavar=("min", "max"),
        help="number of AND gates in AIGER circuit",
    )
    parser.add_argument(
        "--num-ands-plus-latches",
        action=argparse_min_max(),
        nargs="*",
        default=(0, None),
        metavar=("min", "max"),
        help=("number of AND gates plus latches in AIGER" "circuit"),
    )
    parser.add_argument(
        "--num-assumptions",
        action=argparse_min_max(),
        nargs="*",
        default=(0, None),
        metavar=("min", "max"),
        help="number of assumptions",
    )
    parser.add_argument(
        "--num-guarantees",
        action=argparse_min_max(),
        nargs="*",
        default=(1, None),
        metavar=("min", "max"),
        help="number of guarantees",
    )
    parser.add_argument(
        "--num-latches",
        action=argparse_min_max(),
        nargs="*",
        default=(0, None),
        metavar=("min", "max"),
        help="number of latches in AIGER circuit",
    )
    parser.add_argument(
        "--num-vars",
        action=argparse_min_max(),
        nargs="*",
        default=(0, None),
        metavar=("min", "max"),
        help="number of variables in AIGER circuit",
    )
    parser.add_argument("--realizable-frac", default=1.0, type=float)


def argparse_min_max(default_min=0, default_max=None):
    """Action for argparse implementing:
    if no values are given sets min and max to default values,
    if one value is given sets min to that value and max to default value,
    if two values are given sets min to first value and max to second value,
    if more than two values are given raises an ArgumentTypeError.
    """

    class MinMaxAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                setattr(namespace, self.dest, (default_min, default_max))
            elif len(values) == 1:
                setattr(namespace, self.dest, (int(values[0]), default_max))
            elif len(values) == 2:
                setattr(namespace, self.dest, (int(values[0]), int(values[1])))
            else:
                raise argparse.ArgumentTypeError(
                    (
                        "Too many values given to specifiy mininum and maximum"
                        f"for argument {self.dest}"
                    )
                )

    return MinMaxAction


def check_upper_bounds(sample, params, counters):
    violated_bounds = []
    guarantees = sample["guarantees"]
    num_guarantees = len(guarantees)
    if params["num_guarantees"][1] is not None:
        if params["num_guarantees"][1] < num_guarantees:
            violated_bounds.append("max_num_guarantees")
    num_vars, _, num_latches, _, num_ands = aiger.header_ints_from_str(sample["circuit"])
    if params["num_vars"][1] is not None:
        if params["num_vars"][1] < num_vars:
            violated_bounds.append("max_num_vars")
    if params["num_latches"][1] is not None:
        if params["num_latches"][1] < num_latches:
            violated_bounds.append("max_num_latches")
    if params["num_ands"][1] is not None:
        if params["num_ands"][1] < num_ands:
            violated_bounds.append("max_num_ands")
    if params["num_ands_plus_latches"][1] is not None:
        if params["num_ands_plus_latches"][1] < num_ands + num_latches:
            violated_bounds.append("max_num_ands_plus_latches")
    if params["max_frac_ands"] is not None:
        if counters["ands"].get(num_ands, 0) >= params["max_frac_ands"] * params["num_samples"]:
            violated_bounds.append("max_frac_ands")
    return violated_bounds


def check_lower_bounds(sample, params):
    violated_bounds = []
    guarantees = sample["guarantees"]
    num_guarantees = len(guarantees)
    if params["num_guarantees"][0] > num_guarantees:
        violated_bounds.append("min_num_guarantees")
    num_vars, _, num_latches, _, num_ands = aiger.header_ints_from_str(sample["circuit"])
    if params["num_vars"][0] > num_vars:
        violated_bounds.append("min_num_vars")
    if params["num_latches"][0] > num_latches:
        violated_bounds.append("min_num_latches")
    if params["num_ands"][0] > num_ands:
        violated_bounds.append("min_num_ands")
    if params["num_ands_plus_latches"][0] > num_ands + num_latches:
        violated_bounds.append("min_num_ands_plus_latches")
    return violated_bounds


@ray.remote
def csv_dataset_writer(
    queue,
    folder_path: str,
    num_samples: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    curriculum: bool = False,
) -> None:
    filepaths = []
    files = []
    file_writers = []

    for split in ["train", "val", "test"]:
        filepath = os.path.join(folder_path, split + ".csv")
        filepaths.append(filepath)
        file = open(filepath, "w", newline="")
        files.append(file)
        if curriculum:
            fieldnames = ["properties", "type", "inputs", "outputs", "realizable", "circuits"]
        else:
            fieldnames = [
                "assumptions",
                "guarantees",
                "id_DecompLTLSpec",
                "inputs",
                "outputs",
                "realizable",
                "circuit",
                "id_AIGERCircuit",
                "syn_time",
            ]
        file_writer = csv.DictWriter(
            file, extrasaction="ignore", fieldnames=fieldnames, quoting=csv.QUOTE_ALL
        )
        file_writer.writeheader()
        file_writers.append(file_writer)

    file_probs = [train_frac, val_frac, test_frac]
    file_counts = [0, 0, 0]
    file_target_counts = [int(prob * num_samples) for prob in file_probs]

    while sum(file_counts) < num_samples:
        sample = queue.get(block=True)
        index = np.random.choice(range(len(file_probs)), p=file_probs)
        file_writer = file_writers[index]
        if curriculum:
            raise NotImplementedError
            row = curriculum_sample_to_csv_row(sample)
            file_writer.writerow(row)  # TODO dict instead of list
        else:
            file_writer.writerow(sample_to_csv_fields(sample))
        file_counts[index] += 1
        if file_counts[index] == file_target_counts[index]:
            files[index].close()
            logger.info("%d samples written to %s", file_counts[index], filepaths[index])
            file_probs[index] = 0
            if sum(file_probs) > 0:
                # normalize probabilities
                file_probs = [prob * 1 / sum(file_probs) for prob in file_probs]
            else:
                # if rounded down counts are less than number of samples the
                # remaining samples are dropped
                break


@ray.remote
def csv_file_writer(queue, filepath: str, curriculum: bool = False) -> None:
    csv_file = open(filepath, "w", newline="")
    if curriculum:
        fieldnames = ["properties", "type", "inputs", "outputs", "realizable", "circuits"]
    else:
        fieldnames = [
            "assumptions",
            "guarantees",
            "id_DecompLTLSpec",
            "inputs",
            "outputs",
            "realizable",
            "circuit",
            "id_AIGERCircuit",
            "syn_time",
        ]
    file_writer = csv.DictWriter(
        csv_file, extrasaction="ignore", fieldnames=fieldnames, quoting=csv.QUOTE_ALL
    )
    file_writer.writeheader()
    sample = queue.get(block=True)
    while sample:
        if curriculum:
            raise NotImplementedError
            row = curriculum_sample_to_csv_row(sample)
            file_writer.writerow(row)  # TODO dict instead of list
        else:
            file_writer.writerow(sample_to_csv_fields(sample))
        sample = queue.get(block=True)
    csv_file.close()
