""""Verified sample"""

from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

from ...dtypes import DType, ValidationResult
from .beam_search_sample import Beam
from .eval_sample import EvalLabeledSample, EvalSample, LabeledSample

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)
V = TypeVar("V", bound=ValidationResult)


@dataclass(eq=False)
class VerifiedSample(EvalSample[I, T], Generic[I, T, V]):
    verification: V = None
    verification_err: str = None
    verification_time: float = None


@dataclass(eq=False)
class VerifiedLabeledSample(EvalLabeledSample[I, T], Generic[I, T, V]):
    verification: V = None
    verification_err: str = None
    verification_time: float = None


@dataclass(eq=False)
class VerifiedBeam(Beam[T], Generic[T, V]):
    verification: V = None
    verification_err: str = None
    verification_time: float = None


@dataclass(eq=False)
class VerifiedBeamSearchSample(VerifiedSample[I, T, V], Generic[I, T, V]):
    beams: List[VerifiedBeam[T, V]] = field(default_factory=list)

    def add_beam(self, beam: VerifiedBeam[T, V]) -> None:
        self.beams.append(beam)
        if beam.id == 0:
            if self.pred is not None:
                print(self)
                print(beam)
            # assert self.pred is None
            # assert self.pred_enc is None
            # assert self.pred_dec_err is None
            # assert self.verification is None
            self.pred = beam.pred
            self.pred_enc = beam.pred_enc
            self.pred_dec_err = beam.pred_dec_err
            self.verification = beam.verification
            self.time = beam.time
            self.verification_time = beam.verification_time


@dataclass(eq=False)
class VerifiedBeamSearchLabeledSample(
    VerifiedBeamSearchSample[I, T, V], LabeledSample[I, T], Generic[I, T, V]
):
    pass
