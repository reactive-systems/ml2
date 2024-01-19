"""BoSy Ray Worker"""

import logging

import ray

from ...ltl.ltl_spec.decomp_ltl_spec import DecompLTLSpec
from .bosy import BoSy


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
        for problem in problems:
            solution = bosy.synthesize_spec(
                spec=problem.ltl_spec,
                optimize=optimize,
                timeout=timeout,
            )
            problem.ltl_syn_solution = solution

        ray.get(ds_server.post_solved_problems.remote(problems))


@ray.remote
def bosy_worker_fn_dict(
    ds_server,
    cpu_count=2,
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
        for problem_dict in problems:
            problem = DecompLTLSpec.from_dict(problem_dict)
            solution = bosy.synthesize_spec(spec=problem, optimize=optimize, timeout=timeout)
            status = solution.status
            circuit = solution.circuit.to_str()
            time = solution.time
            problem_dict.update({"status": status, "circuit": circuit, "syn_time": time})
        ray.get(ds_server.post_solved_problems.remote(problems))


def add_bosy_args(parser):
    parser.add_argument(
        "--bosy-timeout", type=float, default=10.0, metavar="timeout", help="BoSy timeout"
    )
    parser.add_argument(
        "--bosy-optimize", action="store_true", help="Optimize AIGER w.r.t. number of AND gates"
    )
