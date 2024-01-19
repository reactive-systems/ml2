"""Abstract metric class"""

import json
import os
from abc import abstractmethod
from typing import Any, Dict, List

from ..samples import Sample


class Metric:
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def add(self, sample: Sample) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def add_batch(self, sample_batch: List[Sample]) -> Any:
        for sample in sample_batch:
            self.add(sample)

    @abstractmethod
    def compute(self) -> Any:
        raise NotImplementedError()

    def compute_dict(self) -> Dict[str, Any]:
        return {self.name: self.compute()}

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    def save_to_path(self, path: str) -> None:
        filepath = os.path.join(path, self.name + ".json")
        with open(filepath, "w") as metric_file:
            json.dump(self.compute_dict(), metric_file, indent=2)
