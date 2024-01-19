"""Accuracy per sequence"""

import numpy as np

from ..samples import EvalLabeledSample
from .metric import Metric


class AccPerSeq(Metric):
    def __init__(
        self,
        count_none: bool = True,
        pad_same_length: bool = False,
        pad_id: int = 0,
        name: str = "acc_per_seq",
    ) -> None:
        self.count_none = count_none
        self.pad_same_length = pad_same_length
        self.pad_id = pad_id

        self.acc_not_norm = 0
        self.count = 0
        super().__init__(name=name)

    def add(self, sample: EvalLabeledSample) -> int:
        if sample.pred_enc is None or sample.tar_enc is None:
            if self.count_none:
                self.count += 1
            return 0

        pred_enc = np.array(sample.pred_enc)
        tar_enc = np.array(sample.tar_enc)
        pred_size = pred_enc.size
        tar_size = tar_enc.size

        if self.pad_same_length and pred_size != tar_size:
            if pred_size < tar_size:
                pred_enc = np.pad(
                    pred_enc,
                    (0, (tar_size - pred_size)),
                    mode="constant",
                    constant_values=self.pad_id,
                )
            else:
                tar_enc = np.pad(
                    tar_enc,
                    (0, (pred_size - tar_size)),
                    mode="constant",
                    constant_values=self.pad_id,
                )

        assert pred_enc.size == tar_enc.size

        diff = pred_enc - tar_enc
        if diff.sum() == 0:
            acc = 1
        else:
            acc = 0
        self.acc_not_norm += acc
        self.count += 1
        return acc

    def compute(self) -> float:
        if self.count > 0:
            return self.acc_not_norm / self.count
        return 0.0

    def reset(self) -> None:
        self.acc_not_norm = 0
        self.count = 0
