"""Metric Group"""

from typing import Any, Dict, List

from ..samples import Sample
from .metric import Metric


class MetricGroup(Metric):
    def __init__(self, metrics: List[Metric], name: str = "metrics") -> None:
        self.metrics = metrics
        super().__init__(name=name)

    def add(self, sample: Sample) -> Any:
        for metric in self.metrics:
            metric.add(sample)

    def compute(self) -> Dict[str, Any]:
        result = {}
        for metric in self.metrics:
            result = {**result, **metric.compute_dict()}
        return result

    def compute_dict(self) -> Dict[str, Any]:
        return self.compute()

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
