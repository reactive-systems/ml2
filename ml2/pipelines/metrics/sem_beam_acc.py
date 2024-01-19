"""Semantic accuracy"""

from typing import Dict, Optional

from ..samples import VerifiedBeamSearchSample
from .metric import Metric


class SemBeamAcc(Metric):
    def __init__(
        self,
        count_none: bool = True,
        name: str = "sem_beam_acc",
    ) -> None:
        self.count_success = 0
        self.count_total = 0
        self.count_none = count_none
        super().__init__(name=name)

    def add(self, sample: VerifiedBeamSearchSample) -> Optional[bool]:
        all_beams_none = True
        has_success_beam = False
        for beam in sample.beams:
            if beam.verification is not None and beam.verification.validation_success is not None:
                all_beams_none = False
                has_success_beam = has_success_beam or beam.verification.validation_success

        if all_beams_none and self.count_none:
            self.count_total += 1
            return None
        else:
            self.count_total += 1
            if has_success_beam:
                self.count_success += 1
            return has_success_beam

    def compute_dict(self) -> Dict[str, float]:
        return {
            "semantic_accuracy": self.count_success / self.count_total if self.count_total else 0
        }

    def reset(self) -> None:
        self.count_success = 0
        self.count_total = 0
