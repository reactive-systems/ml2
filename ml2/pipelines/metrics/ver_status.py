"""Verification status metric"""

from typing import Dict

from ..samples import VerifiedSample
from .metric import Metric


class VerStatus(Metric):
    def __init__(
        self,
        count_none: bool = True,
        name: str = "ver_status",
    ) -> None:
        self.count_none = count_none

        self.count_total = 0
        self.status_count: Dict[str, int] = {}
        super().__init__(name=name)

    def add(self, sample: VerifiedSample) -> str:
        if sample.verification is None or sample.verification.validation_status is None:
            if self.count_none:
                self.count_total += 1
                self.status_count["None"] = self.status_count.get("None", 0) + 1
            return "None"
        else:
            self.count_total += 1
        status = sample.verification.validation_status
        self.status_count[status] = self.status_count.get(status, 0) + 1
        return status

    def compute_dict(self) -> Dict[str, float]:
        if self.count_total > 0:
            acc_dict = {s: (c / self.count_total) for s, c in self.status_count.items()}
            return acc_dict
        return {}

    def reset(self) -> None:
        self.count_total = 0
        self.status_count = {}
