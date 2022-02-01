"""A generator that generates synthesis data from a file of guarantees"""

import argparse
import copy
import json
import logging
import os.path
import re
import sys

import numpy as np

import ray
from ray.util.queue import Queue

from ... import aiger
from ...data import add_data_gen_args
from ...tools.bosy import add_bosy_args, bosy_worker_fn
from ...tools.strix import add_strix_args, strix_worker_fn
from ..ltl_spec import LTLSpecPatternData
from .ltl_syn_status import LTLSynStatus
from .ltl_syn_data import LTLSynData, LTLSynSplitData
from .ltl_syn_data_gen_common import (
    check_lower_bounds,
    check_upper_bounds,
    csv_dataset_writer,
    csv_file_writer,
    add_ltl_syn_data_gen_args,
    progress_bar,
    ProgressActor,
)

ray.init(dashboard_host="0.0.0.0")
# ray.init(address='auto')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def translate_substrings(string: str, table: dict):
    if not table:
        return string
    return re.sub(
        "|".join(re.escape(original) for original in table), lambda k: table[k.group(0)], string
    )


@ray.remote
class DataSetActor:
    def __init__(
        self,
        guarantees: list,
        inputs: list,
        outputs: list,
        progress_actor,
        params: dict,
        sample_queue,
        timeouts_queue,
        assumptions: list = None,
    ):
        self.assumptions = assumptions
        self.guarantees = guarantees
        self.inputs = inputs
        self.outputs = outputs
        self.progress_actor = progress_actor
        self.params = params
        self.sample_queue = sample_queue
        self.timeouts_queue = timeouts_queue

        if self.assumptions:
            self.assumptions_ids = list(range(len(self.assumptions)))
        self.guarantees_ids = list(range(len(self.guarantees)))
        self.counters = {"processing": 0, "valid": 0, "realizable": 0, "ands": {}}
        self.prob_realizable = self.params["realizable_frac"]
        self.open = []
        self.sample_ids = []

        logger.setLevel(logging.INFO)

    def register_worker(self):
        self.progress_actor.update.remote("worker")

    def has_unsolved_problems(self):
        return self.counters["valid"] + self.counters["processing"] < self.params["num_samples"]

    def add_assumption(self, problem: dict):
        choices = self.assumptions_ids
        if self.params["unique_assumptions"]:
            choices = list(choices)
            for idx in problem["assumptions_ids"]:
                choices.remove(idx)
        idx = np.random.choice(choices)
        assumption = dict(self.assumptions[idx])
        if self.params["resample_aps"]:
            while True:
                try:
                    resampled_inputs = np.random.choice(
                        problem["inputs"], len(assumption["inputs"]), replace=False
                    ).tolist()
                    resampled_outputs = np.random.choice(
                        problem["outputs"], len(assumption["outputs"]), replace=False
                    ).tolist()
                except ValueError:
                    choices.remove(idx)
                    if choices:
                        idx = np.random.choice(choices)
                    else:
                        logger.error("No assumption patterns left")
                        break
                    assumption = dict(self.assumptions[idx])
                    continue
                translation_table = dict(
                    zip(
                        assumption["inputs"] + assumption["outputs"],
                        resampled_inputs + resampled_outputs,
                    )
                )
                assumption["pattern"] = translate_substrings(
                    assumption["pattern"], translation_table
                )
                break
        problem["assumptions"].append(assumption["pattern"])
        problem["assumptions_ids"].append(idx)
        problem["num_assumption_trials"] += 1
        self.progress_actor.update.remote("assumption_trial")
        return problem

    def add_guarantee(self, problem: dict):
        choices = self.guarantees_ids
        if self.params["unique_guarantees"]:
            # list provides a deep copy
            choices = list(choices)
            for idx in problem["guarantees_ids"]:
                choices.remove(idx)
        idx = np.random.choice(choices)
        # dict provides a deep copy
        guarantee = dict(self.guarantees[idx])
        if self.params["resample_aps"]:
            resampled_inputs = np.random.choice(
                self.inputs, len(guarantee["inputs"]), replace=False
            ).tolist()
            resampled_outputs = np.random.choice(
                self.outputs, len(guarantee["outputs"]), replace=False
            ).tolist()

            translation_table = dict(
                zip(
                    guarantee["inputs"] + guarantee["outputs"],
                    resampled_inputs + resampled_outputs,
                )
            )
            guarantee["pattern"] = translate_substrings(guarantee["pattern"], translation_table)
            guarantee["inputs"] = resampled_inputs
            guarantee["outputs"] = resampled_outputs
        problem["guarantees"].append(guarantee["pattern"])
        logger.debug(problem["guarantees"])
        if self.params["all_aps"]:
            problem["inputs"] = self.inputs
            problem["outputs"] = self.outputs
        else:
            problem["inputs"] = sorted(list(set().union(problem["inputs"], guarantee["inputs"])))
            problem["outputs"] = sorted(
                list(set().union(problem["outputs"], guarantee["outputs"]))
            )
        logger.debug(problem["inputs"])
        logger.debug(problem["outputs"])
        problem["guarantees_ids"].append(idx)
        return problem

    def get_problems(self):
        batch_size = self.params["batch_size"]
        if (
            self.counters["valid"] + self.counters["processing"] + batch_size
            > self.params["num_samples"]
        ):
            batch_size = (
                self.params["num_samples"] - self.counters["valid"] - self.counters["processing"]
            )
        problems = []
        for _ in range(batch_size):
            if self.open:
                parent_problem = self.open.pop(0)
                if parent_problem["status"] == LTLSynStatus.UNREALIZABLE:
                    if parent_problem["num_assumption_trials"] == 0:
                        problem = copy.deepcopy(parent_problem)
                        if not self.params["curriculum"]:
                            parent_problem["parent"] = None
                        problem["unrealizable_parent"] = parent_problem
                        problem["circuit"] = ""
                    else:
                        problem = parent_problem
                        problem["assumptions"].pop()
                        problem["assumptions_ids"].pop()
                    problem = self.add_assumption(problem)
                else:
                    if not self.params["curriculum"]:
                        parent_problem["parent"] = None
                    problem = copy.deepcopy(parent_problem)
                    problem["parent"] = parent_problem
                    problem["unrealizable_parent"] = None
                    problem["circuit"] = ""
                    problem = self.add_guarantee(problem)
                    problem["num_assumption_trials"] = 0
            else:
                problem = {
                    "assumptions": [],
                    "guarantees": [],
                    "inputs": [],
                    "outputs": [],
                    "guarantees_ids": [],
                    "assumptions_ids": [],
                    "num_assumption_trials": 0,
                    "parent": None,
                }
                problem = self.add_guarantee(problem)
            problems.append(problem)
        self.counters["processing"] += batch_size
        return problems

    def post_solved_problems(self, problems: list):
        batch_size = len(problems)
        for problem in problems:
            status = problem["status"]
            num_guarantees = len(problem["guarantees"])
            max_num_guarantees = self.params["num_guarantees"][1]
            num_assumptions = len(problem["assumptions"])
            max_num_assumptions = self.params["num_assumptions"][1]
            num_assumption_trials = problem["num_assumption_trials"]
            max_num_assumption_trials = self.params["max_assumption_trials"]
            if status == LTLSynStatus.REALIZABLE and (
                not max_num_guarantees or num_guarantees < max_num_guarantees
            ):
                problem["realizable"] = 1
                self.open.append(problem)
            elif (
                status == LTLSynStatus.UNREALIZABLE
                and (not max_num_assumptions or num_assumptions < max_num_assumptions)
                and (
                    not max_num_assumption_trials
                    or num_assumption_trials < max_num_assumption_trials
                )
            ):
                problem["realizable"] = 0
                self.open.append(problem)
            elif (
                status == LTLSynStatus.UNREALIZABLE
                or status == LTLSynStatus.TIMEOUT
                or (status == LTLSynStatus.REALIZABLE and num_guarantees == max_num_guarantees)
            ):
                if 0.0 < self.prob_realizable < 1.0:
                    max_num_realizable = int(self.params["num_samples"] * self.prob_realizable)
                    max_num_unrealizable = self.params["num_samples"] - max_num_realizable
                    if self.counters["realizable"] >= max_num_realizable:
                        self.prob_realizable = 0.0
                    if (
                        self.counters["valid"] - self.counters["realizable"]
                        >= max_num_unrealizable
                    ):
                        self.prob_realizable = 1.0
                choose_realizable = np.random.choice(
                    [True, False], p=[self.prob_realizable, 1.0 - self.prob_realizable]
                )
                if status == LTLSynStatus.TIMEOUT:
                    problem["realizable"] = -1
                    self.timeouts_queue.put(problem)
                    self.progress_actor.update.remote("timeout")
                if not choose_realizable and status != LTLSynStatus.UNREALIZABLE:
                    continue
                if not choose_realizable and "unrealizable_parent" in problem:
                    problem = problem["unrealizable_parent"]
                if choose_realizable and status in (
                    LTLSynStatus.UNREALIZABLE,
                    LTLSynStatus.TIMEOUT,
                ):
                    problem = problem["parent"]
                if not problem:
                    continue
                problem["realizable"] = int(choose_realizable)
                try:
                    violated_bounds = check_upper_bounds(problem, self.params, self.counters)
                except ValueError as error:
                    logger.error(
                        (
                            "Checking upper bounds of the following sample failed:"
                            "\n%s\nwith error:\n%s"
                        ),
                        problem,
                        error,
                    )
                    continue
                if violated_bounds:
                    self.progress_actor.update_multi.remote(violated_bounds)
                    self.progress_actor.update.remote("invalid")
                    continue
                try:
                    violated_bounds = check_lower_bounds(problem, self.params)
                except ValueError as error:
                    logger.error(
                        (
                            "Checking lower bounds of the following sample failed:"
                            "\n%s\nwith error:\n%s"
                        ),
                        problem,
                        error,
                    )
                    continue
                if violated_bounds:
                    self.progress_actor.update_multi.remote(violated_bounds)
                    self.progress_actor.update.remote("invalid")
                    continue
                guarantees_ids = problem["guarantees_ids"]
                if self.params["unique_samples"]:
                    guarantees_ids_tuple = tuple(sorted(guarantees_ids))
                    if guarantees_ids_tuple in self.sample_ids:
                        self.progress_actor.update.remote("duplicates")
                        continue
                    else:
                        self.sample_ids.append(guarantees_ids_tuple)
                self.sample_queue.put(problem)
                self.counters["valid"] += 1
                if self.params["max_frac_ands"]:
                    _, _, _, _, num_ands = aiger.header_ints_from_str(problem["circuit"])
                    self.counters["ands"][num_ands] = self.counters["ands"].get(num_ands, 0) + 1
                if choose_realizable:
                    self.counters["realizable"] += 1
                    self.progress_actor.update.remote("realizable")
                self.progress_actor.update.remote("samples")
            elif status == LTLSynStatus.ERROR:
                self.progress_actor.update.remote("error")
                logger.warning("Error occurred for problem %s", problem)
            else:
                logger.warning("Unknown status %s for problem %s", status, problem)
        self.counters["processing"] -= batch_size
        return


def main(args):

    ltl_spec_patterns = LTLSpecPatternData.load(args.patterns)
    guarantees = ltl_spec_patterns.guarantees
    assumptions = ltl_spec_patterns.assumptions

    # create folder and files
    folder_path = LTLSynData.local_path(args.name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        logger.info("Created folder %s", folder_path)

    data_gen_stats_file = os.path.join(folder_path, "data_gen_stats.json")
    flag_filepath = os.path.join(folder_path, "metadata.json")
    args_dict = vars(args)
    with open(flag_filepath, "w") as flag_file:
        json.dump(args_dict, flag_file, indent=2, sort_keys=True)
    logger.info("Command line arguments written to %s", flag_filepath)

    progress_actor = ProgressActor.remote()  # pylint: disable=no-member
    samples_queue = Queue(maxsize=args.num_samples)
    timeouts_queue = Queue(maxsize=args.num_samples)
    # pylint: disable=no-member
    ds_actor = DataSetActor.remote(
        guarantees,
        args.inputs,
        args.outputs,
        progress_actor,
        args_dict,
        samples_queue,
        timeouts_queue,
        assumptions,
    )
    dataset_writer_result = csv_dataset_writer.remote(
        samples_queue,
        folder_path,
        args.num_samples,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.curriculum,
    )
    timeouts_file = os.path.join(folder_path, "timeouts.csv")
    timeouts_writer_result = csv_file_writer.remote(timeouts_queue, timeouts_file, args.curriculum)
    if args.tool == "bosy":
        worker_results = [
            bosy_worker_fn.remote(
                ds_actor,
                id=i,
                optimize=args.bosy_optimize,
                port=50051 + i,
                timeout=args.bosy_timeout,
            )
            for i in range(args.num_workers)
        ]
    elif args.tool == "strix":
        worker_results = [
            strix_worker_fn.remote(
                ds_actor,
                id=i,
                port=50051 + i,
                minimize_aiger=args.strix_auto,
                timeout=args.strix_timeout,
            )
            for i in range(args.num_workers)
        ]
    else:
        sys.exit(f"Unknown synthesis tool {args.tool}")
    progress_bar(progress_actor, args.num_samples, data_gen_stats_file)
    ray.get(worker_results)
    ray.get(dataset_writer_result)
    timeouts_queue.put(None)
    ray.get(timeouts_writer_result)
    split_dataset = LTLSynSplitData.load(args.name)
    # stats = split_dataset.stats(['train', 'val', 'test'])
    # stats_file = os.path.join(folder_path, 'circuit-stats.json')
    # write_stats(stats, stats_file)
    # plot_file = os.path.join(folder_path, 'circuit-stats.png')
    # plot_stats(stats, plot_file)
    split_dataset.shuffle()
    split_dataset.save(
        name=args.name, upload=args.upload, overwrite_local=True, add_to_wandb=args.add_to_wandb
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a synthesis dataset from a set of assumption patterns and a set of guarantee patterns"
    )
    add_data_gen_args(parser)
    add_ltl_syn_data_gen_args(parser)
    add_strix_args(parser)
    add_bosy_args(parser)
    parser.add_argument("--inputs", nargs="*", default=["i0", "i1", "i2", "i3", "i4"])
    parser.add_argument("--outputs", nargs="*", default=["o0", "o1", "o2", "o3", "o4"])
    parser.add_argument(
        "--no-unique-assumptions",
        action="store_false",
        dest="unique_assumptions",
        help=("assumptions in a single sample are not " "necessarily unique"),
    )
    parser.add_argument(
        "--no-unique-guarantees",
        action="store_false",
        dest="unique_guarantees",
        help=("guarantees in a single sample are not " "necessarily unique"),
    )
    parser.add_argument(
        "--no-unique-samples",
        action="store_false",
        dest="unique_samples",
        help="samples in dataset are not necessarily unique",
    )
    parser.add_argument(
        "--max-assumption-trials",
        type=int,
        default=None,
        metavar="max",
        help=(
            "maximum number of trials to find an assumption"
            "that makes the specification realizable"
        ),
    )
    parser.add_argument(
        "-p", "--patterns", type=str, default="scp-0", metavar="name", help=("name of patterns")
    )
    parser.add_argument(
        "--resample-aps", action="store_true", help="resample atomic propositions in guarantees"
    )
    parser.add_argument(
        "--tool", choices=["bosy", "strix"], default="strix", help="synthesis tool"
    )
    main(parser.parse_args())
