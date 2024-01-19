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
from ...tools.bosy import add_bosy_args, bosy_worker_fn
from ...tools.strix import add_strix_args, strix_worker_fn
from ..ltl_spec import DecompLTLSpec
from .ltl_syn_data_gen_common import csv_file_writer
from .ltl_syn_dataset import LTLSynDataset
from .ltl_syn_problem import LTLSynProblem
from .ltl_syn_status import LTLSynStatus

ray.init(dashboard_host="0.0.0.0")
# ray.init(address='auto')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote
class DataSetActor:
    def __init__(
        self,
        specs: Dataset[DecompLTLSpec],
        progress_actor,
        params: dict,
        sample_queue,
        timeouts_queue,
    ):
        self.specs = specs
        self.progress_actor = progress_actor
        self.params = params
        self.sample_queue = sample_queue
        self.timeouts_queue = timeouts_queue

        self.num_specs = self.specs.size
        self.id = 0

        self.counters = {"processing": 0, "realizable": 0, "unrealizable": 0, "timeouts": 0}
        self.open = []
        self.sample_ids = []

        logger.setLevel(logging.INFO)

    def register_worker(self):
        self.progress_actor.update.remote("worker")

    def has_unsolved_problems(self):
        return self.id < self.num_specs or len(self.open) > 0

    def get_problems(self):
        problems = []
        batch_size = 0
        for _ in range(self.params["batch_size"]):
            if self.open:
                spec = self.open.pop(0)
            elif self.id < self.num_specs:
                spec = self.specs[self.id]
                self.progress_actor.update.remote("samples")
                self.id += 1
                if self.params["partial"]:
                    for i in range(len(spec.guarantees)):
                        partial_spec = copy.deepcopy(spec)
                        partial_spec.guarantees.pop(i)
                        self.open.append(partial_spec)
                    spec = self.open.pop(0)
            else:
                break
            problem = LTLSynProblem(ltl_spec=spec, ltl_syn_solution=None)
            problems.append(problem)
            batch_size += 1
        self.counters["processing"] += batch_size
        return problems

    def post_solved_problems(self, problems: list):
        batch_size = len(problems)
        for problem in problems:
            status = problem.ltl_syn_solution.status
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

        self.counters["processing"] -= batch_size
        return


@ray.remote
def csv_file_writer(queue, filepath: str) -> None:
    csv_file = open(filepath, "w", newline="")
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
        file_writer.writerow(sample.to_csv_fields(notation="infix"))
        sample = queue.get(block=True)
    csv_file.close()


def main(args):
    specs = load_dataset(args.specs)
    specs.dtype = DecompLTLSpec
    if args.num_samples is not None:
        specs.sample(args.num_samples)

    # create folder and files
    folder_path = LTLSynDataset.local_path_from_name(args.name, project="ltl-syn")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        logger.info("Created folder %s", folder_path)

    data_gen_stats_file = os.path.join(folder_path, "data_gen_stats.json")
    metadata_filepath = os.path.join(folder_path, "metadata.json")
    args_dict = vars(args)
    filename = args.name.split("/")[-1] + ".csv"
    metadata = {
        "data_type": "DecompLTLSynProblem",
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
    parser.add_argument("--name", type=str, metavar="NAME", required=True, help="dataset name")
    parser.add_argument("-n", "--num-samples", type=int, default=None, help="number of samples")
    parser.add_argument("--partial", action="store_true", help="add partial specs")
    parser.add_argument("-s", "--specs", type=str, default="ltl-syn/scpa-2/test")
    parser.add_argument(
        "--tool", choices=["bosy", "strix"], default="strix", help="synthesis tool"
    )
    parser.add_argument("--num-workers", type=int, default=6, help="number of workers")
    parser.add_argument(
        "-u", "--upload", action="store_true", help="upload generated data to GCP storage bucket"
    )
    main(parser.parse_args())
