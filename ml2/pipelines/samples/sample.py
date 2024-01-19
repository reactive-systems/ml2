"""Sample"""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ...dtypes import DType

T = TypeVar("T", bound=DType)


@dataclass(eq=False)
class Sample(Generic[T]):
    inp: T
    id: int = None
    name: str = None


@dataclass(eq=False)
class EncodedSample(Sample[T]):
    inp_enc: Any = None
    inp_enc_err: str = None
