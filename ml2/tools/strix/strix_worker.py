"""Strix Worker"""

import logging

import ray

from ...ltl import DecompLTLSpec
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem
from .strix import Strix


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
    timeout=None,
):
    strix = Strix(port=port, cpu_count=cpu_count, mem_limit=mem_limit)
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    ds_server.register_worker.remote()
    while ray.get(ds_server.has_unsolved_problems.remote()):
        problems = ray.get(ds_server.get_problems.remote())
        for problem in problems:
            solution = strix.synthesize(ToolLTLSynProblem(parameters={}))
            solution = strix.synthesize_spec(
                spec=problem.ltl_spec,
                minimize_aiger=minimize_aiger,
                minimize_mealy=minimize_mealy,
                timeout=timeout,
            )
            problem.ltl_syn_solution = solution
        ray.get(ds_server.post_solved_problems.remote(problems))


@ray.remote
def strix_worker_fn_dict(
    ds_server,
    cpu_count=2,
    id=0,
    log_level="info",
    mem_limit="2g",
    minimize_aiger=False,
    minimize_mealy=False,
    port=50051,
    timeout=None,
):
    strix = Strix(port=port, cpu_count=cpu_count, mem_limit=mem_limit)
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    ds_server.register_worker.remote()
    while ray.get(ds_server.has_unsolved_problems.remote()):
        problems = ray.get(ds_server.get_problems.remote())
        for problem_dict in problems:
            problem = DecompLTLSpec.from_dict(problem_dict)
            solution = strix.synthesize_spec(
                spec=problem,
                minimize_aiger=minimize_aiger,
                minimize_mealy=minimize_mealy,
                timeout=timeout,
            )
            status = solution.status
            circuit = solution.circuit.to_str()
            time = solution.time
            problem_dict.update({"status": status, "circuit": circuit, "syn_time": time})
        ray.get(ds_server.post_solved_problems.remote(problems))


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
