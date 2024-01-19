"""Null metric"""


from typing import Any, Dict, List

from ..samples import Sample
from .metric import Metric


class NullMetric(Metric):
    def __init__(self, name: str = "null-metric") -> None:
        super().__init__(name=name)

    def add(self, sample: Sample) -> Any:
        pass

    def add_batch(self, sample_batch: List[Sample]) -> Any:
        pass

    def compute(self) -> Any:
        return None

    def compute_dict(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        pass
