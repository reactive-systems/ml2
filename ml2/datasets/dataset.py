"""Abstract dataset class"""

import logging
import os
import os.path
from abc import abstractmethod
from typing import Any, Dict, Generator, Generic, Type, TypeVar

from ..artifact import Artifact
from ..dtypes.dtype import DType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_datasets():
    raise NotImplementedError()


T = TypeVar("T", bound=DType)


class Dataset(Artifact, Generic[T]):
    WANDB_TYPE = "dataset"

    def __init__(
        self,
        name: str,
        dtype: Type[T],
        # shuffle, sample, and save_to are useful for configs
        shuffle: bool = False,
        sample: int = None,
        save_to: str = None,
        **kwargs
    ):
        self.dtype = dtype
        super().__init__(name=name, **kwargs)

        if shuffle:
            self.shuffle()
            self.history["base"] = self.full_name
            self.history["shuffle"] = True
        if sample is not None:
            self.sample(sample)
            self.history["base"] = self.full_name
            self.history["sample"] = sample
        if save_to is not None:
            self.project = None
            self.name = save_to
            self.history = {}
            self.save()

    @abstractmethod
    def add_sample(self, sample: T, **kwargs) -> None:
        """Add sample to the dataset"""
        raise NotImplementedError()

    def config_postprocessors(self) -> list:
        def postprocess_sample(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("sample", None)
            annotations.pop("sample", None)

        def postprocess_save_to(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("save_to", None)
            annotations.pop("save_to", None)

        def postprocess_shuffle(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("shuffle", None)
            annotations.pop("shuffle", None)

        return [
            postprocess_sample,
            postprocess_save_to,
            postprocess_shuffle,
        ] + super().config_postprocessors()

    @abstractmethod
    def generator(self, **kwargs) -> Generator[T, None, None]:
        """Yields samples of data type T"""
        raise NotImplementedError()

    @abstractmethod
    def sample(self, n: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def shuffle(self, seed: int = None) -> None:
        """Shuffles the dataset"""
        raise NotImplementedError()

    @property
    def size(self) -> int:
        """Size of dataset"""
        raise NotImplementedError()

    def stats(self, **kwargs) -> Any:
        """Statistics of the dataset"""
        raise NotImplementedError()

    @property
    def stats_path(self) -> str:
        return os.path.join(self.local_path, "stats")
