"""Decentralized resolution data generation"""

import argparse
import json
import logging
import os
import signal
import time

import ray
from ray.util.queue import Queue

from ...data_gen import (
    DataServer,
    ProgressActor,
    add_dist_data_gen_args,
    data_writing_progress_bar,
)
from ...datasets import CSVDatasetWriter, SplitDatasetWriter, load_dataset
from ...tools.booleforce import BooleForce
from ..prop_sat_status import PropSatStatus
from .cnf_res_problem import CNFResProblem
from .neurosat_data_gen_common import add_neurosat_data_gen_args
from .res_data_gen_common import CNFResDataGenProblem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TimeoutException(Exception):
    pass


@ray.remote
def resolution_data_gen_worker_fn(
    server: DataServer,
    batch_size: int,
    min_n: int,
    max_n: int,
    p_k_2: float,
    p_geo: float,
    timeout: float,
    id: int,
    port: int,
    mem_limit: str,
) -> None:
    booleforce = BooleForce(port=port, mem_limit=mem_limit)

    def signal_handler(num, stack):
        raise TimeoutException()

    while not ray.get(server.finished.remote()):
        batch = []
        problem = None
        while len(batch) < batch_size:
            if problem is None:
                problem = CNFResDataGenProblem.from_random(
                    min_n=min_n, max_n=max_n, timeout=timeout
                )
                problem = problem.add_clause(p_k_2=p_k_2, p_geo=p_geo)

            # wrap booleforce call in signal timeout block because in rare cases it does not respond for unknown reasons
            # in these cases the grpc server seems still alive but does not respond to requests anymore
            signal.signal(signal.SIGALRM, signal_handler)
            signal_timeout = int(timeout) + 1
            signal.alarm(signal_timeout)
            try:
                problem.solution = booleforce.check_sat(
                    formula=problem.formula, timeout=problem.timeout
                )
            except TimeoutException:
                logger.error(
                    f"BooleForce did not respond after {signal_timeout} seconds "
                    f"for problem {problem.formula.to_str()}"
                )
                logger.info("Deleting BooleForce")
                booleforce.__del__()
                logger.info("Restarting BooleForce")
                booleforce = BooleForce(port=port)
                problem = None
                continue
            finally:
                signal.alarm(0)

            if problem.solution.status == PropSatStatus("sat"):
                problem = problem.add_clause(p_k_2=p_k_2, p_geo=p_geo)
            elif problem.solution.status == PropSatStatus("unsat"):
                batch.append(problem)
                problem = None
            else:
                problem = None

        ray.get(server.post_problem_batch.remote(batch))


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
    samples_queue = Queue(maxsize=args.num_workers * args.batch_size)
    # pylint: disable=no-member
    ds_actor = DataServer.remote(
        num_samples=args.num_samples,
        progress_actor=progress_actor,
        sample_queue=samples_queue,
    )
    worker_results = []
    for i in range(args.num_workers):
        worker_results.append(
            resolution_data_gen_worker_fn.remote(
                server=ds_actor,
                batch_size=args.batch_size,
                min_n=args.min_n,
                max_n=args.max_n,
                p_k_2=args.p_k_2,
                p_geo=args.p_geo,
                timeout=args.timeout,
                id=i,
                port=50051 + i,
                mem_limit=args.mem_lim_workers,
            )
        )
        time.sleep(args.sleep_workers)

    data_writing_progress_bar(
        dataset_writer, progress_actor, samples_queue, args.num_samples, data_gen_stats_file
    )
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
