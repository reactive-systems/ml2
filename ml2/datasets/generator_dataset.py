"""Abstract generator dataset class"""

import logging
from abc import abstractmethod
from typing import Any, Generator, Generic, Type, TypeVar

from ..dtypes.dtype import DType
from .dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_datasets():
    raise NotImplementedError()


T = TypeVar("T", bound=DType)


# Dateset that only consist of a generator.
class GeneratorDataset(Dataset, Generic[T]):
    WANDB_TYPE = "dataset"

    def __init__(
        self,
        name: str,
        dtype: Type[T],
        generator: Generator[T, None, None],
        project: str = None,
        auto_version: bool = False,
        metadata: dict = None,
    ):
        self._generator = generator
        super().__init__(
            name=name,
            dtype=dtype,
            project=project,
            auto_version=auto_version,
            metadata=metadata,
        )

    def add_sample(self, sample: T, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def generator(self, **kwargs) -> Generator[T, None, None]:
        return self._generator

    @property
    def metadata(self) -> dict:
        metadata = super().metadata
        metadata["dtype"] = self.dtype.__name__
        return metadata

    def sample(self, n: int) -> None:
        raise NotImplementedError()

    def save_to_path(self, path: str) -> None:
        raise NotImplementedError()

    def shuffle(self, seed: int = None) -> None:
        raise NotImplementedError()

    @property
    def size(self) -> int:
        raise NotImplementedError()

    def stats(self, **kwargs) -> Any:
        raise NotImplementedError()
