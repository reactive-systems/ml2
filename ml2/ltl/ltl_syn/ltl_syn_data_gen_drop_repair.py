"""Data generation that synthesizes a dataset of specifications"""

import argparse
import copy
import csv
import json
import logging
import os.path
import sys

import ray
from ray.util.queue import Queue

from ...data_gen import ProgressActor, progress_bar
from ...datasets import Dataset, load_dataset
from ...dtypes import CSV
from ...tools.bosy import add_bosy_args, bosy_worker_fn
from ...tools.strix import add_strix_args, strix_worker_fn
from ..ltl_spec import DecompLTLSpec
from .decomp_ltl_syn_problem import DecompLTLSynProblem
from .ltl_syn_dataset import LTLSynDataset
from .ltl_syn_problem import LTLSynSolution
from .ltl_syn_status import LTLSynStatus

ray.init(dashboard_host="0.0.0.0")
# ray.init(address='auto')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PartialSpecRepairProblem(CSV):
    def __init__(
        self,
        ltl_spec: DecompLTLSpec,
        ltl_syn_solution: LTLSynSolution,
        full_ltl_spec: DecompLTLSpec,
        full_ltl_syn_solution: LTLSynSolution,
        hash: str,
    ) -> None:
        self.ltl_spec = ltl_spec
        self.ltl_syn_solution = ltl_syn_solution
        self.full_ltl_spec = full_ltl_spec
        self.full_ltl_syn_solution = full_ltl_syn_solution
        self.hash = hash


@ray.remote
class DataSetActor:
    def __init__(
        self,
        data: Dataset[DecompLTLSynProblem],
        progress_actor,
        params: dict,
        sample_queue,
        timeouts_queue,
    ):
        self.data = data
        self.progress_actor = progress_actor
        self.params = params
        self.sample_queue = sample_queue
        self.timeouts_queue = timeouts_queue

        self.num_samples = self.data.size
        self.id = 0

        self.counters = {"processing": 0, "realizable": 0, "unrealizable": 0, "timeouts": 0}

        logger.setLevel(logging.INFO)

    def register_worker(self):
        self.progress_actor.update.remote("worker")

    def has_unsolved_problems(self):
        return self.id < self.num_samples

    def get_problems(self):
        problems = []
        batch_size = 0
        for _ in range(self.params["batch_size"]):
            if self.id < self.num_samples:
                sample = self.data[self.id]
                self.id += 1
                full_spec = sample.ltl_spec
                full_solution = sample.ltl_syn_solution
                partial_spec = copy.deepcopy(full_spec)
                partial_spec.guarantees.pop()
                if len(partial_spec.guarantees) > 1:
                    partial_spec.guarantees.pop()

                problem = PartialSpecRepairProblem(
                    ltl_spec=partial_spec,
                    ltl_syn_solution=None,
                    full_ltl_spec=full_spec,
                    full_ltl_syn_solution=full_solution,
                    hash=sample.hash,
                )
            else:
                break
            problems.append(problem)
            batch_size += 1
        self.counters["processing"] += batch_size
        return problems

    def post_solved_problems(self, problems: list):
        batch_size = len(problems)
        for problem in problems:
            status = problem.ltl_syn_solution.status
            if (
                problem.ltl_syn_solution.circuit.to_str()
                == problem.full_ltl_syn_solution.circuit.to_str()
            ):
                self.progress_actor.update.remote("equal")
                if self.params["filter_equals"]:
                    self.progress_actor.update.remote("samples")
                    continue
            if status == LTLSynStatus("realizable"):
                self.progress_actor.update.remote("realizable")
                self.sample_queue.put(problem)
            elif status == LTLSynStatus("unrealizable"):
                self.progress_actor.update.remote("unrealizable")
                self.sample_queue.put(problem)
            elif status == LTLSynStatus("timeout"):
                self.timeouts_queue.put(problem)
                self.progress_actor.update.remote("timeout")
            elif status == LTLSynStatus("error"):
                self.progress_actor.update.remote("error")
                logger.warning("Error occurred for problem %s", problem)
            else:
                logger.warning("Unknown status %s for problem %s", status, problem)

            self.progress_actor.update.remote("samples")
        self.counters["processing"] -= batch_size
        return


@ray.remote
def csv_file_writer(queue, filepath: str) -> None:
    csv_file = open(filepath, "w", newline="")
    fieldnames = [
        "status",
        "assumptions",
        "guarantees",
        "repair_circuit",
        "inputs",
        "outputs",
        "realizable",
        "circuit",
        "hash",
        "partial_assumptions",
        "partial_guarantees",
    ]
    file_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    file_writer.writeheader()
    sample = queue.get(block=True)
    while sample:
        full_spec_fields = sample.full_ltl_spec.to_csv_fields(notation="infix")
        full_solution_fields = sample.full_ltl_syn_solution.to_csv_fields()
        partial_spec_fields = sample.ltl_spec.to_csv_fields(notation="infix")
        partial_solution_fields = sample.ltl_syn_solution.to_csv_fields()
        row = {
            "status": "changed",
            "assumptions": full_spec_fields["assumptions"],
            "guarantees": full_spec_fields["guarantees"],
            "repair_circuit": partial_solution_fields["circuit"],
            "inputs": full_spec_fields["inputs"],
            "outputs": full_spec_fields["outputs"],
            "realizable": full_solution_fields["realizable"],
            "circuit": full_solution_fields["circuit"],
            "hash": sample.hash,
            "partial_assumptions": partial_spec_fields["assumptions"],
            "partial_guarantees": partial_spec_fields["guarantees"],
        }
        file_writer.writerow(row)
        sample = queue.get(block=True)
    csv_file.close()


def main(args):
    specs = load_dataset(args.dataset)
    if args.num_samples is not None:
        specs.sample(args.num_samples)

    # create folder and files
    folder_path = LTLSynDataset.local_path_from_name(args.name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        logger.info("Created folder %s", folder_path)

    data_gen_stats_file = os.path.join(folder_path, "data_gen_stats.json")
    metadata_filepath = os.path.join(folder_path, "metadata.json")
    args_dict = vars(args)
    filename = args.name.split("/")[-1] + ".csv"
    metadata = {
        "filename": filename,
        "name": args.name,
        "type": "CSVDataset",
        "data_gen_args": args_dict,
    }
    with open(metadata_filepath, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
    logger.info("Command line arguments written to %s", metadata_filepath)

    progress_actor = ProgressActor.remote()  # pylint: disable=no-member
    samples_queue = Queue(maxsize=specs.size)
    timeouts_queue = Queue(maxsize=specs.size)
    # pylint: disable=no-member
    ds_actor = DataSetActor.remote(
        specs,
        progress_actor,
        args_dict,
        samples_queue,
        timeouts_queue,
    )
    dataset_file = os.path.join(folder_path, filename)
    dataset_writer_result = csv_file_writer.remote(
        samples_queue,
        dataset_file,
    )
    timeouts_file = os.path.join(folder_path, "timeouts.csv")
    timeouts_writer_result = csv_file_writer.remote(timeouts_queue, timeouts_file)
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
    progress_bar(progress_actor, specs.size, data_gen_stats_file)
    ray.get(worker_results)
    samples_queue.put(None)
    ray.get(dataset_writer_result)
    timeouts_queue.put(None)
    ray.get(timeouts_writer_result)
    # split_dataset = LTLSynDataset.load(args.name, project="ltl-syn")
    # split_dataset.shuffle()
    # split_dataset.save(
    #     name=args.name, upload=args.upload, overwrite_local=True, add_to_wandb=args.add_to_wandb
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a synthesis dataset from a set of specifications"
    )
    add_strix_args(parser)
    add_bosy_args(parser)

    parser.add_argument(
        "--add-to-wandb", action="store_true", help="add data to Weights and Biases"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="size of batches provided to worker"
    )
    parser.add_argument("--filter-equals", action="store_true", dest="filter_equals")
    parser.add_argument(
        "--name",
        default="ltl-repair/ps-0/test",
        type=str,
        help="dataset name",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=None, help="number of samples")
    parser.add_argument("-d", "--dataset", type=str, default="ltl-repair/scpa-2/test")
    parser.add_argument(
        "--tool", choices=["bosy", "strix"], default="strix", help="synthesis tool"
    )
    parser.add_argument("--num-workers", type=int, default=6, help="number of workers")
    parser.add_argument(
        "-u", "--upload", action="store_true", help="upload generated data to GCP storage bucket"
    )
    main(parser.parse_args())
