"""Beam search sample"""

from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

from ...dtypes import DType
from ...tokenizers import TFEncoding
from .eval_sample import EvalLabeledSample, EvalSample

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)


@dataclass(eq=False)
class Beam(Generic[T]):
    id: int
    pred: T = None
    pred_enc: TFEncoding = None
    pred_dec_err: str = None
    time: float = None


@dataclass(eq=False)
class BeamSearchSample(EvalSample[I, T], Generic[I, T]):
    beams: List[Beam[T]] = field(default_factory=list)

    def add_beam(self, beam: Beam[T]) -> None:
        self.beams.append(beam)
        if beam.id == 0:
            if self.pred is not None:
                print(self)
                print(beam)
            # assert self.pred is None
            # assert self.pred_enc is None
            # assert self.pred_dec_err is None
            self.pred = beam.pred
            self.pred_enc = beam.pred_enc
            self.pred_dec_err = beam.pred_dec_err
            self.time = beam.time


@dataclass(eq=False)
class BeamSearchLabeledSample(BeamSearchSample[I, T], EvalLabeledSample[I, T], Generic[I, T]):
    pass
