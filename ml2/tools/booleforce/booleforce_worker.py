"""BooleForce worker"""

import ray

from ...data_gen import DataGenServer
from .booleforce import BooleForce


@ray.remote
def booleforce_worker_fn(
    server: DataGenServer,
    id: int,
    port: int,
    mem_limit: str,
) -> None:
    booleforce = BooleForce(port=port, mem_limit=mem_limit)
    server.register_worker.remote()
    while True:
        problems = ray.get(server.get_problem_batch.remote())
        if problems is None:
            break
        for problem in problems:
            solution = booleforce.check_sat(formula=problem.formula, timeout=problem.timeout)
            problem.solution = solution
        ray.get(server.post_problem_batch.remote(problems))
    server.deregister_worker.remote()
