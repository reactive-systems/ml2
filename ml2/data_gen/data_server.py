""""Data server that gets problems from workers and sends them to the queue"""

from typing import Any, List

import ray
from ray.util.queue import Queue

from .progress_actor import ProgressActor


@ray.remote
class DataServer(object):
    def __init__(
        self,
        num_samples: int,
        progress_actor: ProgressActor = None,
        sample_queue: Queue = None,
    ):
        self.num_samples = num_samples
        self.progress_actor = progress_actor
        self.sample_queue = sample_queue

        self.progress_actor.update.remote("samples", 0)

    def finished(self) -> bool:
        return ray.get(self.progress_actor.get.remote("samples")) >= self.num_samples

    def post_problem(self, problem: Any) -> None:
        self.sample_queue.put(problem, block=True)
        self.progress_actor.update.remote("samples")

    def post_problem_batch(self, batch: List[Any]) -> None:
        for problem in batch:
            if not self.finished():
                self.post_problem(problem)
