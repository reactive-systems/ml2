""""Evaluated sample"""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ...dtypes import DType
from .labeled_sample import LabeledSample
from .sample import EncodedSample

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)


@dataclass(eq=False)
class EvalSample(EncodedSample[I], Generic[I, T]):
    pred: T = None
    pred_enc: Any = None
    pred_dec_err: str = None
    time: float = None


@dataclass(eq=False)
class EvalLabeledSample(EvalSample[I, T], LabeledSample[I, T], Generic[I, T]):
    pass
