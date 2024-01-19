"""Metric counting encoding and decoding errors"""

from typing import Any, Dict

from ..samples import (
    EncodedSample,
    EvalLabeledSample,
    LabeledSample,
    VerifiedLabeledSample,
    VerifiedSample,
)
from .metric import Metric


class ErrCounter(Metric):
    def __init__(self, name: str = "err-counter") -> None:
        self.num_inp_enc_errs = 0
        self.num_samples = 0
        super().__init__(name=name)

    def add(self, sample: EncodedSample) -> bool:
        self.num_samples += 1
        if sample.inp_enc_err is not None:
            self.num_inp_enc_errs += 1
            return True
        return False

    def compute(self) -> int:
        return self.num_inp_enc_errs

    def compute_dict(self) -> Dict[str, int]:
        return {"inp_enc_errs": self.compute()}

    def reset(self) -> None:
        self.num_inp_enc_errs = 0
        self.num_samples = 0


class SupervisedErrCounter(ErrCounter):
    def __init__(self, name: str = "supervised-err-counter") -> None:
        self.num_tar_enc_errs = 0
        super().__init__(name=name)

    def add(self, sample: LabeledSample) -> bool:
        inp_err = super().add(sample)
        if inp_err:
            return inp_err
        if sample.tar_enc_err is not None:
            self.num_tar_enc_errs += 1
            return True
        return False

    def compute_dict(self) -> Dict[str, Any]:
        errs = super().compute_dict()
        errs["tar_enc_errs"] = self.num_tar_enc_errs
        return errs

    def reset(self) -> None:
        super().reset()
        self.num_tar_enc_errs = 0


class EvalErrCounter(ErrCounter):
    def __init__(self, name: str = "eval-err-counter") -> None:
        self.num_pred_dec_errs = 0
        super().__init__(name)

    def add(self, sample: EvalLabeledSample) -> bool:
        tar_err = super().add(sample)
        if tar_err:
            return tar_err
        if sample.pred_dec_err is not None:
            self.num_pred_dec_errs += 1
            return True
        return False

    def compute_dict(self) -> Dict[str, Any]:
        errs = super().compute_dict()
        errs["pred_dec_errs"] = self.num_pred_dec_errs
        return errs

    def reset(self) -> None:
        super().reset()
        self.num_pred_dec_errs = 0


class EvalSupervisedErrCounter(SupervisedErrCounter):
    def __init__(self, name: str = "eval-supervised-err-counter") -> None:
        self.num_pred_dec_errs = 0
        super().__init__(name)

    def add(self, sample: EvalLabeledSample) -> bool:
        tar_err = super().add(sample)
        if tar_err:
            return tar_err
        if sample.pred_dec_err is not None:
            self.num_pred_dec_errs += 1
            return True
        return False

    def compute_dict(self) -> Dict[str, Any]:
        errs = super().compute_dict()
        errs["pred_dec_errs"] = self.num_pred_dec_errs
        return errs

    def reset(self) -> None:
        super().reset()
        self.num_pred_dec_errs = 0


class VerificationErrCounter(EvalErrCounter):
    def __init__(self, name: str = "verified-err-counter") -> None:
        self.num_verification_errs = 0
        super().__init__(name)

    def add(self, sample: VerifiedSample) -> bool:
        eval_err = super().add(sample)
        if eval_err:
            return eval_err
        if sample.verification_err is not None:
            self.num_verification_errs += 1
            return True
        return False

    def compute_dict(self) -> Dict[str, Any]:
        errs = super().compute_dict()
        errs["verification_errs"] = self.num_verification_errs
        return errs

    def reset(self) -> None:
        super().reset()
        self.num_verification_errs = 0


class VerificationSupervisedErrCounter(EvalSupervisedErrCounter):
    def __init__(self, name: str = "verified-supervised-err-counter") -> None:
        self.num_verification_errs = 0
        super().__init__(name)

    def add(self, sample: VerifiedLabeledSample) -> bool:
        eval_err = super().add(sample)
        if eval_err:
            return eval_err
        if sample.verification_err is not None:
            self.num_verification_errs += 1
            return True
        return False

    def compute_dict(self) -> Dict[str, Any]:
        errs = super().compute_dict()
        errs["verification_errs"] = self.num_verification_errs
        return errs

    def reset(self) -> None:
        super().reset()
        self.num_verification_errs = 0
