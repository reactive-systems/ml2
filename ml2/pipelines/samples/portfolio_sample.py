"""Portfolio sample"""

from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar

from ...dtypes import DType
from .sample import Sample

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)


@dataclass(eq=False)
class Result(Generic[T]):
    id: int
    result: T
    name: Optional[str] = None
    time: Optional[float] = None


@dataclass(eq=False)
class PortfolioSample(Sample[I], Generic[I, T]):
    results: List[Result[T]] = field(default_factory=list)

    def add_result(
        self, result: T, name: Optional[str] = None, time: Optional[float] = None
    ) -> None:
        self.results.append(Result(id=len(self.results), result=result, name=name, time=time))
