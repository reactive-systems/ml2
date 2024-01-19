"""Metric counting the number of samples"""

from typing import Dict

from ..samples import Sample
from .metric import Metric


class Counter(Metric):
    def __init__(self, name: str = "counter") -> None:
        self.num_samples = 0
        super().__init__(name=name)

    def add(self, sample: Sample) -> bool:
        self.num_samples += 1

    def compute(self) -> int:
        return self.num_samples

    def compute_dict(self) -> Dict[str, int]:
        return {"num_samples": self.compute()}

    def reset(self) -> None:
        self.num_samples = 0
