"""Split dataset writer"""

import logging
from typing import Any, Dict, Generic, List, TypeVar

import numpy as np

from ..artifact import Artifact
from ..dtypes.dtype import DType
from .dataset_writer import DatasetWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


T = TypeVar("T", bound=DType)
DWA = DatasetWriter[T]
DW = TypeVar("DW", bound=DWA)


class SplitDatasetWriter(Artifact, Generic[T, DW]):
    def __init__(
        self,
        name: str,
        splits: Dict[str, DW],
        target_sizes: Dict[str, int],
        project: str = None,
        metadata: dict = None,
    ):
        assert len(splits) > 0
        self.splits = splits
        for ts in target_sizes.values():
            assert ts >= 0
        self.target_sizes = target_sizes
        self.total_target_size = sum(self.target_sizes.values())
        super().__init__(
            name=name,
            project=project,
            metadata=metadata,
        )

    @property
    def split_names(self) -> List[str]:
        return [*self.splits]

    @property
    def split_probs(self) -> Dict[str, float]:
        total_size = sum([split.size() for split in self.splits.values()])
        if total_size < self.total_target_size:
            return {
                name: (target_size - self.splits[name].size())
                / (self.total_target_size - total_size)
                for name, target_size in self.target_sizes.items()
            }
        else:
            return {name: 0.0 for name in self.split_names}

    def add_sample(self, sample: T, **kwargs) -> None:
        assert any(self.split_probs.values())
        name = np.random.choice(
            self.split_names, p=[self.split_probs[name] for name in self.split_names]
        )
        split = self.splits[name]
        split.add_sample(sample, **kwargs)
        if split.size() == self.target_sizes[name]:
            logger.info("%d samples added to %s", split.size(), split.full_name)

    def close(self) -> None:
        for split in self.splits.values():
            split.close()

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config["dtype"] = self.splits[self.split_names[0]].dtype.__name__

        def postprocess_splits(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config["splits"] = {k: w.full_name for k, w in self.splits.items()}

        def postprocess_target_sizes(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("target_sizes", None)
            annotations.pop("target_sizes", None)

        def postprocess_type(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config["type"] = "SplitDataset"

        return [
            postprocess_dtype,
            postprocess_splits,
            postprocess_target_sizes,
            postprocess_type,
        ] + super().config_postprocessors()

    def save(
        self,
        add_to_wandb: bool = False,
        overwrite_bucket: bool = False,
        overwrite_local: bool = False,
        recurse: bool = False,
        upload: bool = False,
    ) -> None:
        super().save(
            add_to_wandb=add_to_wandb,
            overwrite_bucket=overwrite_bucket,
            overwrite_local=overwrite_local,
            recurse=recurse,
            upload=upload,
        )

        if recurse:
            for dw in self.splits.values():
                dw.save(
                    add_to_wandb=add_to_wandb,
                    overwrite_bucket=overwrite_bucket,
                    overwrite_local=overwrite_local,
                    recurse=recurse,
                    upload=upload,
                )
