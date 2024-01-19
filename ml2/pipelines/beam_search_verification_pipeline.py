""""Beam search verification pipeline"""

import logging
import time
from dataclasses import asdict
from typing import Generator, Generic, Optional, TypeVar

from ..dtypes import DType, ValidationResult
from ..registry import register_type
from .metrics import (
    Acc,
    AccPerSeq,
    Counter,
    EvalSupervisedErrCounter,
    Metric,
    MetricGroup,
    SemBeamAcc,
    VerificationErrCounter,
    VerificationSupervisedErrCounter,
    VerStatus,
)
from .samples import (
    BeamSearchLabeledSample,
    BeamSearchSample,
    VerifiedBeam,
    VerifiedBeamSearchLabeledSample,
    VerifiedBeamSearchSample,
)
from .verification_pipeline import VerificationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)
V = TypeVar("V", bound=ValidationResult)


@register_type
class BeamSearchVerificationPipeline(VerificationPipeline[I, T], Generic[I, T]):
    def eval(self, *args, **kwargs) -> Generator[VerifiedBeamSearchSample[I, T, V], None, None]:
        return super().eval(*args, **kwargs)  # type: ignore

    def eval_sample(self, *args, **kwargs) -> VerifiedBeamSearchSample[I, T, V]:
        return super().eval_sample(*args, **kwargs)  # type: ignore

    def eval_supervised_sample(self, *args, **kwargs) -> VerifiedBeamSearchLabeledSample[I, T, V]:
        return super().eval_supervised_sample(*args, **kwargs)  # type: ignore

    def eval_supervised(
        self, *args, **kwargs
    ) -> Generator[VerifiedBeamSearchLabeledSample[I, T, V], None, None]:
        return super().eval_supervised(*args, **kwargs)  # type: ignore

    def verify_sample(self, sample: BeamSearchSample[I, T]) -> VerifiedBeamSearchSample[I, T, V]:
        ver_sample = VerifiedBeamSearchSample(
            inp=sample.inp,
            id=sample.id,
            inp_enc=sample.inp_enc,
            inp_enc_err=sample.inp_enc_err,
        )
        for beam in sample.beams:
            start = time.time()
            try:
                verification: Optional[ValidationResult] = (
                    self.verifier.verify(sample.inp, beam.pred)
                    if sample.pred is not None
                    else None
                )
                verification_err = None
                end = time.time()
            except Exception as err:
                verification = None
                verification_err = str(err)
                end = time.time()

            beam.time = (beam.time + (end - start)) if beam.time is not None else None
            ver_sample.add_beam(
                VerifiedBeam(
                    **asdict(beam),
                    verification=verification,
                    verification_err=verification_err,
                    verification_time=end - start
                )
            )
        return ver_sample

    def verify_supervised_sample(
        self, sample: BeamSearchLabeledSample[I, T]
    ) -> VerifiedBeamSearchLabeledSample[I, T, V]:
        ver_sample = VerifiedBeamSearchLabeledSample(
            inp=sample.inp,
            inp_enc=sample.inp_enc,
            inp_enc_err=sample.inp_enc_err,
            id=sample.id,
            tar=sample.tar,
            tar_enc=sample.tar_enc,
            tar_enc_err=sample.tar_enc_err,
        )
        for beam in sample.beams:
            start = time.time()
            try:
                verification: Optional[ValidationResult] = (
                    self.verifier.verify(sample.inp, beam.pred)
                    if sample.pred is not None
                    else None
                )
                verification_err = None
                end = time.time()
            except Exception as err:
                verification = None
                verification_err = str(err)
                end = time.time()

            beam.time = (beam.time + (end - start)) if beam.time is not None else None
            ver_sample.add_beam(
                VerifiedBeam(
                    **asdict(beam),
                    verification=verification,
                    verification_err=verification_err,
                    verification_time=end - start
                )
            )
        return ver_sample

    @staticmethod
    def default_metric() -> Metric:
        return MetricGroup([Counter(), SemBeamAcc(), VerificationErrCounter(), VerStatus()])

    @staticmethod
    def default_supervised_metric() -> Metric:
        return MetricGroup(
            [
                Acc(pad_same_length=True),
                AccPerSeq(pad_same_length=True),
                Counter(),
                EvalSupervisedErrCounter(),
                SemBeamAcc(),
                VerificationSupervisedErrCounter(),
                VerStatus(),
            ]
        )
