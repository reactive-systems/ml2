""""Supervised sample"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ...dtypes import DType
from .sample import EncodedSample

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)


def tar_err():
    raise ValueError("Target is not specified")


@dataclass(eq=False)
class LabeledSample(EncodedSample[I], Generic[I, T]):
    tar: T = field(default_factory=lambda x: tar_err())
    tar_enc: Any = None
    tar_enc_err: str = None
