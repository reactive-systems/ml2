"""Split dataset"""

import logging
from typing import Any, Dict, Generator, Generic, List, Type, TypeVar

from ..dtypes import DType
from ..gcp_bucket import path_exists
from ..registry import register_type
from .dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=DType)
DTA = Dataset[T]
DT = TypeVar("DT", bound=DTA)


@register_type
class SplitDataset(Dataset[T], Generic[T, DT]):
    WANDB_TYPE = "split_dataset"

    def __init__(self, name: str, dtype: Type[T], splits: Dict[str, DT] = None, **kwargs):
        self._splits = splits if splits else {}
        super().__init__(name=name, dtype=dtype, **kwargs)

    def __contains__(self, item):
        return item in self.split_names

    def __delitem__(self, key):
        del self._splits[key]

    def __getitem__(self, name: str) -> DT:
        if name not in self.split_names:
            raise ValueError(
                f"Split {name} does not match any of the available splits {self.split_names}"
            )
        return self._splits[name]

    def __setitem__(self, name: str, split: DT) -> None:
        if split in self.split_names:
            logger.warning("Overwriting split %s", name)
        self._splits[name] = split

    def add_sample(self, sample: T, split: str = None, **kwargs) -> None:
        if split is None:
            raise ValueError("Specifiy split")
        self[split].add_sample(sample, **kwargs)

    def generator(self, splits: List[str] = None, **kwargs) -> Generator[T, None, None]:
        """Yields dataset samples"""
        split_names = splits if splits else self.split_names
        for name in split_names:
            split = self[name]
            for sample in split.generator(**kwargs):
                yield sample

    def items(self):
        return self._splits.items()

    def keys(self):
        return self._splits.keys()

    def values(self):
        return self._splits.values()

    def sample(self, n: int) -> None:
        total_size = self.size
        for split in self.values():
            frac = split.size / total_size
            split.sample(n=int(frac * n))

    def save(
        self,
        add_to_wandb: bool = False,
        overwrite_bucket: bool = False,
        overwrite_local: bool = False,
        recurse: bool = False,
        upload: bool = False,
        **kwargs,
    ) -> None:
        # hack that overwrites if path does not exist because the first save uploads nested splits
        split_to_overwrite_bucket = {}
        for name in self.split_names:
            if not path_exists(self[name].bucket_path):
                split_to_overwrite_bucket[name] = True
            else:
                split_to_overwrite_bucket[name] = overwrite_bucket

        super().save(
            add_to_wandb=add_to_wandb,
            overwrite_bucket=overwrite_bucket,
            overwrite_local=overwrite_local,
            upload=upload,
        )

        if recurse:
            for name in self.split_names:
                self[name].save(
                    add_to_wandb=add_to_wandb,
                    overwrite_bucket=split_to_overwrite_bucket[name],
                    overwrite_local=overwrite_local,
                    upload=upload,
                )

    def shuffle(self, splits: List[str] = None) -> None:
        for name in splits if splits else self.split_names:
            self[name].shuffle()

    @property
    def size(self) -> int:
        """Sum of all split sizes"""
        return sum([split.size for split in self.values()])

    @property
    def split_sizes(self) -> Dict[str, int]:
        """Size of each split"""
        return {name: split.size for name, split in self.items()}

    @property
    def split_names(self) -> List[str]:
        return list(self.keys())

    def split_stats(self, **kwargs) -> Dict[str, Any]:
        """Statistics of each split"""
        return {name: split.stats(**kwargs) for name, split in self.items()}
