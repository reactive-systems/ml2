"""TensorFlow pipeline"""

import logging
import os
from abc import abstractmethod
from typing import Generic, List, TypeVar

import tensorflow as tf

from ..datasets import Dataset
from ..dtypes import DType
from ..registry import register_type
from .model_pipeline import ModelPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=DType)


@register_type
class TFPipeline(ModelPipeline[T], Generic[T]):
    WANDB_TYPE = "pipeline"

    def __init__(self, **kwargs) -> None:
        self._train_model = None
        self._eval_model = None

        super().__init__(**kwargs)

    def default_train_metrics(self) -> List[tf.keras.metrics.Metric]:
        raise NotImplementedError()

    @property
    def eval_model(self):
        if not self._eval_model:
            self._eval_model = self.init_model(training=False)
            logger.info("Created evaluation model")
            # for compatibility with older versions check if path is a directory and retrieve checkpoint via latest_checkpoint function
            if os.path.isdir(self.checkpoint_path):
                checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                if checkpoint:
                    logger.info("Found legacy format checkpoint %s", checkpoint)
                    self._eval_model.load_weights(checkpoint).expect_partial()
                    logger.info("Loaded weights from legacy format checkpoint")
            elif os.path.isfile(self.checkpoint_path):
                logger.info("Found checkpoint %s", self.checkpoint_path)
                self._eval_model.load_weights(self.checkpoint_path)
                logger.info("Loaded weights from checkpoint")
        return self._eval_model

    @property
    def train_model(self):
        if not self._train_model:
            self._train_model = self.init_model(training=True)
            logger.info("Created training model")
            # for compatibility with older versions check if path is a directory and retrieve checkpoint via latest_checkpoint function
            if os.path.isdir(self.checkpoint_path):
                checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                if checkpoint:
                    logger.info("Found legacy format checkpoint %s", checkpoint)
                    self._train_model.load_weights(checkpoint).expect_partial()
                    logger.info("Loaded weights from legacy format checkpoint")
            elif os.path.isfile(self.checkpoint_path):
                logger.info("Found checkpoint %s", self.checkpoint_path)
                self._train_model.load_weights(self.checkpoint_path)
                logger.info("Loaded weights from checkpoint")
        return self._train_model

    @abstractmethod
    def build_tf_dataset(self, dataset: Dataset[T]) -> tf.data.Dataset:
        raise NotImplementedError()
