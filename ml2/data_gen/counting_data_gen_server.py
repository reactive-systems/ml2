"""Abstract counting data generation server class"""

from typing import Any, List, Optional

import ray
from ray.util.queue import Queue

from .data_gen_server import DataGenServer
from .progress_actor import ProgressActor


class CountingDataGenServer(DataGenServer):
    def __init__(
        self,
        num_samples: int,
        progress_actor: ProgressActor,
        batch_size: int = 1,
        sample_queue: Queue = None,
    ):
        super().__init__(
            batch_size=batch_size,
            progress_actor=progress_actor,
            sample_queue=sample_queue,
        )

        self.num_samples = num_samples

        self.progress_actor.update.remote("samples", 0)
        self.progress_actor.update.remote("processing", 0)

    def has_problems(self) -> bool:
        finished = ray.get(self.progress_actor.get.remote("samples"))
        processing = ray.get(self.progress_actor.get.remote("processing"))
        return finished + processing < self.num_samples

    def get_problem_batch(self) -> Optional[List[Any]]:
        if not self.has_problems():
            return None
        batch = super().get_problem_batch()
        if batch is not None:
            self.progress_actor.update.remote("processing", len(batch))
        return batch

    def post_problem_batch(self, batch: List[Any]) -> None:
        super().post_problem_batch(batch)
        self.progress_actor.update.remote("processing", -len(batch))
