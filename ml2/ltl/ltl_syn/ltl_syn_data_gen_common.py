"""Common functionality for synthesis data generation"""

import argparse
from asyncio import Event
import csv
import json
import logging
import os
from tqdm import tqdm

import numpy as np
import ray

from ... import aiger
from .ltl_syn_data import curriculum_sample_to_csv_row, sample_to_csv_row

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def add_ltl_syn_data_gen_args(parser):
    parser.add_argument(
        "--add-to-wandb", action="store_true", help="add data to Weights and Biases"
    )
    parser.add_argument(
        "--all-aps",
        action="store_true",
        help=("circuits incorporate all AP contained in the set" "of guarantees"),
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="size of batches provided to worker"
    )
    parser.add_argument(
        "-c", "--curriculum", action="store_true", help="generate data for curriculum learning"
    )
    parser.add_argument(
        "--max-frac-ands",
        type=float,
        default=None,
        metavar="max",
        help=("maximal fraction of circuits with" "0,1,2,... ANG gates"),
    )
    parser.add_argument("--name", type=str, metavar="NAME", required=True, help="dataset name")
    parser.add_argument("-n", "--num-samples", type=int, default=100, help="number of samples")
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
    parser.add_argument("--num-workers", type=int, default=6, help="number of workers")
    parser.add_argument("--realizable-frac", default=1.0, type=float)
    parser.add_argument(
        "-u", "--upload", action="store_true", help="upload generated data to GCP storage bucket"
    )


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
        file_writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if curriculum:
            file_writer.writerow(
                ["properties", "type", "inputs", "outputs", "realizable", "circuits"]
            )
        else:
            file_writer.writerow(
                ["assumptions", "guarantees", "inputs", "outputs", "realizable", "circuit"]
            )
        file_writers.append(file_writer)

    file_probs = [train_frac, val_frac, test_frac]
    file_counts = [0, 0, 0]
    file_target_counts = [int(prob * num_samples) for prob in file_probs]

    while sum(file_counts) < num_samples:
        sample = queue.get(block=True)
        index = np.random.choice(range(len(file_probs)), p=file_probs)
        file_writer = file_writers[index]
        if curriculum:
            row = curriculum_sample_to_csv_row(sample)
        else:
            row = sample_to_csv_row(sample)
        file_writer.writerow(row)
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
    file_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    if curriculum:
        file_writer.writerow(["properties", "type", "inputs", "outputs", "realizable", "circuits"])
    else:
        file_writer.writerow(
            ["assumptions", "guarantees", "inputs", "outputs", "realizable", "circuit"]
        )
    sample = queue.get(block=True)
    while sample:
        if curriculum:
            row = curriculum_sample_to_csv_row(sample)
        else:
            row = sample_to_csv_row(sample)
        file_writer.writerow(row)
        sample = queue.get(block=True)
    csv_file.close()


def progress_bar(progress_actor, num_samples, stats_filepath=None):
    pbar = tqdm(total=num_samples, desc="Generated samples", unit="sample")
    progress_actor.update.remote("samples", 0)
    while True:
        progress = ray.get(progress_actor.wait_for_update.remote())
        pbar.update(progress["samples"] - pbar.n)
        postfix_dict = dict(progress)
        postfix_dict.pop("samples", None)
        pbar.set_postfix(postfix_dict)
        if progress["samples"] >= num_samples:
            if stats_filepath:
                with open(stats_filepath, "w") as stats_file:
                    progress["elapsed"] = pbar.format_dict["elapsed"]
                    json.dump(progress, stats_file, indent=2)
            pbar.close()
            return


@ray.remote
class ProgressActor:
    def __init__(self):
        self.progress = {}
        self.event = Event()

    def update(self, key, delta=1):
        if key in self.progress:
            self.progress[key] += delta
            self.event.set()
        else:
            self.progress[key] = delta
            self.progress = {key: self.progress[key] for key in sorted(self.progress.keys())}
            self.event.set()

    def update_multi(self, keys, delta=1):
        for key in keys:
            self.update(key, delta)

    async def wait_for_update(self):
        await self.event.wait()
        self.event.clear()
        return self.progress
