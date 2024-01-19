"""Abstract Callback class"""


from abc import abstractmethod
from typing import Any, List

from ...artifact import Artifact
from ..samples import Sample


class Callback(Artifact):
    @abstractmethod
    def add(self, sample: Sample, **kwargs) -> Any:
        raise NotImplementedError()

    def add_batch(self, sample_batch: List[Sample], **kwargs) -> Any:
        for sample in sample_batch:
            self.add(sample, **kwargs)
