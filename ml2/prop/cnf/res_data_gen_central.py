"""Centralized resolution data generation"""

import argparse
import json
import logging
import os
import time
from typing import List, Optional

import ray
from ray.util.queue import Queue

from ...data_gen import (
    CountingDataGenServer,
    ProgressActor,
    add_dist_data_gen_args,
    progress_bar,
    progress_bar_init,
)
from ...datasets import CSVDatasetWriter, SplitDatasetWriter, load_dataset
from ...tools.booleforce import booleforce_worker_fn
from ..prop_sat_status import PropSatStatus
from .cnf_res_problem import CNFResProblem
from .neurosat_data_gen_common import add_neurosat_data_gen_args
from .res_data_gen_common import CNFResDataGenProblem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@ray.remote
class ResolutionDataGenServer(CountingDataGenServer):
    def __init__(
        self,
        min_n: int,
        max_n: int,
        p_k_2: float,
        p_geo: float,
        progress_actor: ProgressActor,
        timeout: float = None,
        batch_size: int = 1,
        num_samples: int = 1000,
        sample_queue: Queue = None,
    ) -> None:
        self.min_n = min_n
        self.max_n = max_n
        self.p_k_2 = p_k_2
        self.p_geo = p_geo
        self.timeout = timeout

        self.open_problems: List[CNFResDataGenProblem] = []

        super().__init__(
            batch_size=batch_size,
            num_samples=num_samples,
            progress_actor=progress_actor,
            sample_queue=sample_queue,
        )

    def get_problem(self) -> Optional[CNFResDataGenProblem]:
        if self.open_problems:
            problem = self.open_problems.pop()
        else:
            problem = CNFResDataGenProblem.from_random(
                min_n=self.min_n, max_n=self.max_n, timeout=self.timeout
            )

        return problem.add_clause(p_k_2=self.p_k_2, p_geo=self.p_geo)

    def post_problem(self, problem: CNFResDataGenProblem) -> None:
        self.progress_actor.update.remote(problem.solution.status.value)
        if problem.solution.status == PropSatStatus("sat"):
            self.open_problems.append(problem)
        if problem.solution.status == PropSatStatus("unsat"):
            self.sample_queue.put(problem)
            self.progress_actor.update.remote("samples")


def main(args):
    split_writers = {}
    split_sizes = {}
    for split, f in [("train", args.train_frac), ("val", args.val_frac), ("test", args.test_frac)]:
        split_writers[split] = CSVDatasetWriter(
            name=f"{args.name}/{split}",
            dtype=CNFResProblem,
            header=["formula", "res_proof"],
            filename=split + ".csv",
            project=args.project,
        )
        split_sizes[split] = int(f * args.num_samples)

    dataset_writer = SplitDatasetWriter(
        name=args.name, splits=split_writers, target_sizes=split_sizes, project=args.project
    )

    data_gen_stats_file = os.path.join(dataset_writer.local_path, "data_gen_stats.json")
    data_gen_args_filepath = os.path.join(dataset_writer.local_path, "data_gen_args.json")
    args_dict = vars(args)
    with open(data_gen_args_filepath, "w") as flag_file:
        json.dump(args_dict, flag_file, indent=2, sort_keys=True)
    logger.info("Command line arguments written to %s", data_gen_args_filepath)

    progress_actor = ProgressActor.remote()  # pylint: disable=no-member
    samples_queue = Queue(maxsize=args.num_samples)
    # pylint: disable=no-member
    pbar = progress_bar_init(progress_actor, args.num_samples)
    ds_actor = ResolutionDataGenServer.remote(
        min_n=args.min_n,
        max_n=args.max_n,
        p_k_2=args.p_k_2,
        p_geo=args.p_geo,
        progress_actor=progress_actor,
        timeout=args.timeout,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        sample_queue=samples_queue,
    )
    worker_results = []
    for i in range(args.num_workers):
        worker_results.append(
            booleforce_worker_fn.remote(
                ds_actor,
                id=i,
                port=50051 + i,
                mem_limit=args.mem_lim_workers,
            )
        )
        time.sleep(args.sleep_workers)
    progress_bar(pbar, progress_actor, args.num_samples, data_gen_stats_file)
    counter = 0
    while counter < args.num_samples:
        sample = samples_queue.get(block=True)
        dataset_writer.add_sample(sample)
        counter += 1
    ray.get(worker_results)
    dataset_writer.close()
    dataset_writer.save(recurse=True)

    ds = load_dataset(name=args.name, project=args.project)
    if args.shuffle:
        ds.shuffle()
    ds.save(upload=args.upload, overwrite_local=True, add_to_wandb=args.add_to_wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a resolution proof dataset")
    add_dist_data_gen_args(parser)
    parser.set_defaults(project="prop-res")
    add_neurosat_data_gen_args(parser)
    main(parser.parse_args())
