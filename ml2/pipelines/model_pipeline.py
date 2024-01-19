"""Abstract model pipeline class"""

import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Generic, TypeVar

from ..configurable import Configurable
from ..dtypes import DType
from ..gcp_bucket import download_path
from .pipeline import Pipeline

T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPipeline(Pipeline[T], Generic[T]):
    def __init__(self, model_config: Configurable, checkpoint_name: str = None, **kwargs) -> None:
        self.model_config = model_config

        super().__init__(**kwargs)

        self.checkpoint_name = (
            checkpoint_name if checkpoint_name is not None else self.full_name + "/ckpts"
        )

    @property
    def checkpoint_path(self) -> str:
        return self.local_path_from_name(name=self.checkpoint_name)

    @property
    @abstractmethod
    def eval_model(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def train_model(self):
        raise NotImplementedError()

    @abstractmethod
    def init_model(self, training: bool = False, **kwargs):
        raise NotImplementedError()

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_checkpoint(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            checkpoint_name = config.get("checkpoint_name", None)
            if checkpoint_name is not None:
                checkpoint_path = cls.local_path_from_name(name=checkpoint_name)
                if not os.path.exists(checkpoint_path):
                    bucket_path = cls.bucket_path_from_name(name=checkpoint_name)
                    logger.info("Downloading checkpoint %s", bucket_path)
                    download_path(bucket_path=bucket_path, local_path=checkpoint_path)
                    logger.info("Downloaded checkpoint to %s", checkpoint_path)

        return super().config_preprocessors() + [preprocess_checkpoint]
