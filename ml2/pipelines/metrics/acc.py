"""Accuracy"""

from typing import Tuple

import numpy as np

from ..samples import EvalLabeledSample
from .metric import Metric


class Acc(Metric):
    def __init__(self, pad_same_length: bool = False, pad_id: int = 0, name: str = "acc") -> None:
        self.pad_same_length = pad_same_length
        self.pad_id = pad_id
        self.acc_not_norm = 0.0
        self.count = 0
        super().__init__(name=name)

    def add(self, sample: EvalLabeledSample) -> float:
        pred_enc, tar_enc = np.array(sample.pred_enc), np.array(sample.tar_enc)
        pred_enc, tar_enc = self._pad_enc(pred_enc, tar_enc)

        assert pred_enc.size == tar_enc.size
        sample_acc_not_norm, sample_count = self._sample_acc_and_count(pred_enc, tar_enc)
        return sample_acc_not_norm / sample_count

    def _pad_enc(self, pred_enc: np.ndarray, tar_enc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred_size, tar_size = pred_enc.size, tar_enc.size

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

        return pred_enc, tar_enc

    def _sample_acc_and_count(
        self, pred_array: np.ndarray, tar_array: np.ndarray
    ) -> Tuple[int, int]:
        elem_acc = pred_array == tar_array
        sample_acc_not_norm = elem_acc.sum()
        sample_count = elem_acc.size
        self.acc_not_norm += sample_acc_not_norm
        self.count += sample_count
        return sample_acc_not_norm, sample_count

    def compute(self) -> float:
        if self.count > 0:
            return self.acc_not_norm / self.count
        return 0.0

    def reset(self) -> None:
        self.acc_not_norm = 0.0
        self.count = 0
