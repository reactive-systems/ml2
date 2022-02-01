"""A generator that generates a json file of LTL patterns based on a grammar"""

import argparse
from asyncio import Event
from itertools import product, permutations
import json
import logging
import os.path
import random
from tqdm import tqdm

import ray

from ....data.utils import int_to_abbrev_str
from ....tools import strix
from ...ltl_syn import LTLSynStatus
from .ltl_spec_pattern_grammar import LTLSpecPatternGrammar

ray.init(dashboard_host="0.0.0.0")


@ray.remote
class StatisticsActor:
    def __init__(self):
        self.realizable = 0
        self.unrealizable = 0
        self.worker = 0
        self.event = Event()

    def incr_realizable(self):
        self.realizable += 1
        self.event.set()

    def incr_unrealizable(self):
        self.unrealizable += 1
        self.event.set()

    def incr_worker(self):
        self.worker += 1
        self.event.set()

    async def wait_for_update(self):
        await self.event.wait()
        self.event.clear()
        return {
            "realizable": self.realizable,
            "unrealizable": self.unrealizable,
            "worker": self.worker,
        }


@ray.remote
class Server:
    def __init__(
        self,
        grammar,
        inputs,
        outputs,
        num_samples,
        stats_actor,
        batch_size=1,
        negated_aps=True,
        unique=True,
        verbosity=0,
    ):
        logging.basicConfig(level=verbosity)
        self.patterns = grammar.derive_all()
        self.inputs = inputs
        self.outputs = outputs
        self.negated_aps = negated_aps
        if self.negated_aps:
            negated_inputs = [f"! {i}" for i in self.inputs]
            self.inputs.extend(negated_inputs)
            negated_outputs = [f"! {o}" for o in self.outputs]
            self.outputs.extend(negated_outputs)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.unique = unique
        self.processing = 0
        self.realizable_count = 0
        self.indices = list(range(len(self.patterns)))
        self.pattern_fills = {}
        self.samples = []
        self.stats_actor = stats_actor

    def register_worker(self):
        self.stats_actor.incr_worker.remote()

    def has_unsolved_problems(self):
        return self.realizable_count + self.processing < self.num_samples

    def get_problems(self):
        batch_size = self.batch_size
        if self.realizable_count + self.processing + batch_size > self.num_samples:
            batch_size = self.num_samples - self.realizable_count - self.processing
        problems = []
        for _ in range(batch_size):
            index = random.choice(self.indices)
            pattern = self.patterns[index]
            num_inputs = pattern.num_inputs
            num_outputs = pattern.num_outputs
            if index not in self.pattern_fills:
                self.pattern_fills[index] = list(
                    product(
                        permutations(self.inputs, num_inputs),
                        permutations(self.outputs, num_outputs),
                    )
                )
            fill_choices = self.pattern_fills[index]
            if not fill_choices:
                self.indices.remove(index)
                continue
            pattern_inputs, pattern_outputs = random.choice(fill_choices)
            if self.unique:
                self.pattern_fills[index].remove((pattern_inputs, pattern_outputs))
            guarantee = pattern.fill(pattern_inputs, pattern_outputs)
            specification = {
                "guarantees": [guarantee],
                "inputs": list(set([i.replace("! ", "") for i in pattern_inputs])),
                "outputs": list(set([o.replace("! ", "") for o in pattern_outputs])),
            }
            problems.append(specification)
        self.processing += batch_size
        return problems

    def post_solved_problems(self, problems):
        batch_size = len(problems)
        for problem in problems:
            status = problem["status"]
            pattern = {
                "pattern": problem["guarantees"][0],
                "inputs": problem["inputs"],
                "outputs": problem["outputs"],
            }
            if status == LTLSynStatus.REALIZABLE:
                self.samples.append(pattern)
                self.realizable_count += 1
                self.stats_actor.incr_realizable.remote()
            elif status == LTLSynStatus.UNREALIZABLE:
                self.stats_actor.incr_unrealizable.remote()
            else:
                logging.warning("status of problem %s is %s", problem, status)
        self.processing -= batch_size

    def get_dataset(self):
        return self.samples


def progress_bar(stats_actor, num_samples):
    pbar = tqdm(total=num_samples, desc="Sampled patterns", unit="sample")
    while True:
        stats = ray.get(stats_actor.wait_for_update.remote())
        pbar.update(stats["realizable"] - pbar.n)
        pbar.set_postfix(unrealizable=stats["unrealizable"], worker=stats["worker"])
        if stats["realizable"] >= num_samples:
            pbar.close()
            return


def generate(args):
    grammar = LTLSpecPatternGrammar()
    # pylint: disable=no-member
    stats_actor = StatisticsActor.remote()
    server = Server.remote(
        grammar,
        args.inputs,
        args.outputs,
        args.num_samples,
        stats_actor,
        args.batch_size,
        args.negated_aps,
        args.unique,
        args.verbosity,
    )
    results = [strix.strix_worker.strix_worker.remote(server, args.strix_bin) for id in range(5)]
    progress_bar(stats_actor, args.num_samples)
    ray.get(results)
    return ray.get(server.get_dataset.remote())


def get_folder_name(args):
    folder_name = "n" + int_to_abbrev_str(args.num_samples)
    if args.unique:
        folder_name += "-ug"
    return folder_name


def add_parser_args(parser):
    parser.add_argument("--batch-size", default=10, help="size of batches provided to worker")
    parser.add_argument(
        "--negated-aps",
        default=True,
        help=("when sampling atomic propositions include " "negated atomic propositions"),
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["i0", "i1", "i2", "i3", "i4"],
        help="list of input atomic propositions",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="number of samples")
    parser.add_argument(
        "--outputs",
        nargs="*",
        default=["o0", "o1", "o2", "o3", "o4"],
        help="list of output atomic propositions",
    )
    parser.add_argument(
        "--repo-path", type=str, default="/DeepLogic", help="path to DeepLogic repository"
    )
    parser.add_argument("--unique", type=bool, default=True, help="sampled guarantees are unique")
    parser.add_argument("--verbosity", default=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a dataset of guarantees")
    add_parser_args(parser)
    strix.strix_wrapper.add_parser_args(parser)
    args = parser.parse_args()
    samples = generate(args)
    bucket_file_dir = f"data/synthesis/guarantees/grammar/{get_folder_name(args)}"
    file_dir = f"{args.repo_path}/{bucket_file_dir}"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    filepath = os.path.join(file_dir, "guarantees.json")
    with open(filepath, "w") as f:
        json.dump({"patterns": samples}, f, indent=2)
    logging.info("Successfully written %d patterns to file %s", len(samples), filepath)
