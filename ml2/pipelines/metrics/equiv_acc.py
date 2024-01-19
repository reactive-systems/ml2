"""Equivalence accuracy"""

from typing import Dict, Optional

from ..samples import VerifiedSample
from .metric import Metric


class EquivAcc(Metric):
    def __init__(
        self,
        count_none: bool = True,
        name: str = "equiv_acc",
    ) -> None:

        self.count_success = 0
        self.count_total = 0
        self.count_none = count_none
        super().__init__(name=name)

    def add(self, sample: VerifiedSample) -> Optional[bool]:
        if sample.verification is None or sample.verification.equiv is None:
            if self.count_none:
                self.count_total += 1
            return None
        else:
            self.count_total += 1
            if sample.verification.equiv:
                self.count_success += 1
            return sample.verification.equiv

    def compute_dict(self) -> Dict[str, float]:
        return {"equiv_acc": self.count_success / self.count_total if self.count_total else 0}

    def reset(self) -> None:
        self.count_success = 0
        self.count_total = 0
