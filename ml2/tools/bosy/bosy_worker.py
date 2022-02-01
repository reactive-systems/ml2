"""BoSy Ray Worker"""
import logging
import time

import ray

from ...ltl import LTLSpec
from .bosy import BoSy
from .bosy_wrapper import bosy_wrapper_dict


@ray.remote
def bosy_worker(server, bosy_path, bosy_timeout, temp_dir):
    server.register_worker.remote()
    while ray.get(server.has_unsolved_problems.remote()):
        problems = ray.get(server.get_problems.remote())
        for problem in problems:
            solution = bosy_wrapper_dict(problem, bosy_path, bosy_timeout, temp_dir)
            problem.update(solution)
        ray.get(server.post_solved_problems.remote(problems))


@ray.remote
def bosy_worker_fn(
    ds_server,
    cpu_count=1,
    id=0,
    log_level="info",
    mem_limit="2g",
    optimize=False,
    port=50051,
    timeout=None,
):
    bosy = BoSy(port=port, cpu_count=cpu_count, mem_limit=mem_limit)
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    ds_server.register_worker.remote()
    while ray.get(ds_server.has_unsolved_problems.remote()):
        problems = ray.get(ds_server.get_problems.remote())
        start = time.time()
        for problem_dict in problems:
            problem = LTLSpec.from_dict(problem_dict)
            status, circuit = bosy.synthesize(problem, optimize, timeout)
            problem_dict.update({"status": status, "circuit": circuit})
        end = time.time()
        ray.get(ds_server.post_solved_problems.remote(problems))
        print(f"Worker {id} solved {len(problems)} problems in {end - start} seconds")


def add_bosy_args(parser):
    parser.add_argument(
        "--bosy-timeout", type=float, default=10.0, metavar="timeout", help="BoSy timeout"
    )
    parser.add_argument(
        "--bosy-optimize", action="store_true", help="Optimize AIGER w.r.t. number of AND gates"
    )
