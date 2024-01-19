"""Abstract data generation server class"""

from typing import Any, List, Optional

from ray.util.queue import Queue

from .progress_actor import ProgressActor


class DataGenServer(object):
    def __init__(
        self,
        batch_size: int = 1,
        progress_actor: ProgressActor = None,
        sample_queue: Queue = None,
    ):
        self.batch_size = batch_size
        self.progress_actor = progress_actor
        self.sample_queue = sample_queue

    def get_problem(self) -> Optional[Any]:
        raise NotImplementedError()

    def get_problem_batch(self) -> Optional[List[Any]]:
        batch = []
        for _ in range(self.batch_size):
            p = self.get_problem()
            if p is None:
                break
            batch.append(p)
        if batch == []:
            return None
        return batch

    def post_problem(self, problem: Any) -> None:
        raise NotImplementedError()

    def post_problem_batch(self, batch: List[Any]) -> None:
        for problem in batch:
            self.post_problem(problem)

    def deregister_worker(self):
        self.progress_actor.update.remote("worker", -1)

    def register_worker(self):
        self.progress_actor.update.remote("worker")
