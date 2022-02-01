"""Strix Worker"""

import logging

import ray

from ...ltl import LTLSpec
from .strix import Strix
from .strix_wrapper import strix_wrapper_dict


@ray.remote
def strix_worker(
    server,
    minimize_aiger=False,
    minimize_mealy=False,
    threads=None,
    timeout=None,
    log_level="info",
    id=0,
):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    server.register_worker.remote()
    while ray.get(server.has_unsolved_problems.remote()):
        problems = ray.get(server.get_problems.remote())
        for problem in problems:
            solution = strix_wrapper_dict(
                problem, minimize_aiger, minimize_mealy, threads, timeout
            )
            problem.update(solution)
        ray.get(server.post_solved_problems.remote(problems))


@ray.remote
def strix_worker_fn(
    ds_server,
    cpu_count=2,
    id=0,
    log_level="info",
    mem_limit="2g",
    minimize_aiger=False,
    minimize_mealy=False,
    port=50051,
    threads=1,
    timeout=None,
):
    strix = Strix(port=port, cpu_count=cpu_count, mem_limit=mem_limit)
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    ds_server.register_worker.remote()
    while ray.get(ds_server.has_unsolved_problems.remote()):
        problems = ray.get(ds_server.get_problems.remote())
        # start = time.time()
        for problem_dict in problems:
            problem = LTLSpec.from_dict(problem_dict)
            status, circuit = strix.synthesize(
                problem, minimize_aiger, minimize_mealy, threads, timeout
            )
            problem_dict.update({"status": status, "circuit": circuit})
        # end = time.time()
        ray.get(ds_server.post_solved_problems.remote(problems))
        # print(
        #     f'Worker {id} solved {len(problems)} problems in {end - start} seconds'
        # )


def add_strix_args(parser):
    parser.add_argument(
        "--strix-no-auto",
        action="store_false",
        dest="strix_auto",
        help="no additional minimization of Mealy machine",
    )
    parser.add_argument(
        "--strix-timeout", type=float, default=10.0, metavar="timeout", help="Strix timeout"
    )
