"""Keras trainer"""

import json
import logging
import os
from typing import Any, Dict, List

import tensorflow as tf
import wandb
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from wandb.keras import WandbMetricsLogger

from ..datasets import Dataset
from ..optim.tf_optim import load_tf_optimizer_from_config, tf_optimizer_to_config
from ..pipelines import TFPipeline
from ..registry import register_type
from ..utils.tf_utils import (
    str_to_tf_float_dtype,
    str_to_tf_int_dtype,
    tf_float_dtype_to_str,
    tf_int_dtype_to_str,
)
from .trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHECKPOINT_FILENAME = "ckpt"


class AdamOptimizerArgs:
    def __init__(self, constant_learning_rate: float = None) -> None:
        self.constant_learning_rate = constant_learning_rate


@register_type
class KerasTrainer(Trainer):
    def __init__(
        self,
        pipeline: TFPipeline,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        batch_size: int = 32,
        cache_dataset: bool = True,
        callbacks: List[Callback] = None,
        checkpoint_monitor: str = "val_loss",
        drop_batch_remainder: bool = True,
        dtype_float: tf.DType = tf.float32,
        dtype_int: tf.DType = tf.int32,
        initial_steps: int = 0,
        optimizer: Optimizer = None,
        save_to_wandb: bool = False,
        shuffle_on_load: bool = True,
        steps: int = 100,
        tf_shuffle_buffer_size: int = 0,
        val_freq: int = 10,
        **kwargs,
    ):
        super().__init__(pipeline=pipeline, **kwargs)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if shuffle_on_load:
            self.train_dataset.shuffle()
            logger.info("Shuffled train dataset")
            if self.val_dataset is not None:
                self.val_dataset.shuffle()
                logger.info("Shuffled val dataset")

        self.batch_size = batch_size
        self.cache_dataset = cache_dataset
        self.callbacks = callbacks if callbacks else []
        self.checkpoint_monitor = checkpoint_monitor
        self.drop_batch_remainder = drop_batch_remainder
        self.dtype_float = dtype_float
        self.dtype_int = dtype_int
        self.initial_steps = initial_steps
        self.optimizer = optimizer
        self.save_to_wandb = save_to_wandb
        self.shuffle_on_load = shuffle_on_load
        self.steps = steps
        self.tf_shuffle_buffer_size = tf_shuffle_buffer_size
        self.val_freq = val_freq
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_path, CHECKPOINT_FILENAME),
            monitor=self.checkpoint_monitor,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.local_path)
        self.callbacks += [checkpoint_callback, tensorboard_callback]

        logger.info(
            "Trainer configured with:\n%s",
            "\n".join([f"{a}: {v}" for a, v in sorted(self.get_config().items())]),
        )

        if os.path.exists(self.local_path):
            logger.info("Found existing trainer directory")
        else:
            # create trainer directory
            os.makedirs(self.local_path)
            logger.info("Created trainer directory %s", self.local_path)

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_callbacks(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("callbacks", None)
            annotations.pop("callbacks", None)

        def postprocess_optimizer(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "optimizer"
            if name in config and not isinstance(config[name], str):
                config[name] = tf_optimizer_to_config(config[name])

        def postprocess_tf_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "dtype_float"
            if name in config and isinstance(config[name], tf.DType):
                config[name] = tf_float_dtype_to_str[config[name]]

        def postprocess_tf_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "dtype_int"
            if name in config and isinstance(config[name], tf.DType):
                config[name] = tf_int_dtype_to_str[config[name]]

        return [
            postprocess_callbacks,
            postprocess_optimizer,
            postprocess_tf_float_dtype,
            postprocess_tf_int_dtype,
        ] + super().config_postprocessors()

    def get_tf_dataset(self, dataset: Dataset) -> tf.data.Dataset:
        tf_dataset, error_callbacks = self.pipeline.convert_sl_dataset_to_tf(
            dataset, return_error_callbacks=True
        )
        self.callbacks.extend(error_callbacks)
        if self.cache_dataset:
            tf_dataset = tf_dataset.cache()
        if self.tf_shuffle_buffer_size:
            tf_dataset = tf_dataset.shuffle(
                self.tf_shuffle_buffer_size, reshuffle_each_iteration=False
            )
        tf_dataset = tf_dataset.batch(self.batch_size, drop_remainder=self.drop_batch_remainder)
        tf_dataset = tf_dataset.prefetch(2)
        return tf_dataset

    def train(self):
        super().train()

        if self.stream_to_wandb:
            self.callbacks += [WandbMetricsLogger(log_freq=self.log_freq)]

        train_model = self.pipeline.train_model
        train_model.compile(optimizer=self.optimizer)
        logger.info("Compiled training model")

        tf_train_data = self.get_tf_dataset(self.train_dataset).repeat()
        tf_val_data = self.get_tf_dataset(self.val_dataset)

        history = train_model.fit(
            tf_train_data,
            callbacks=self.callbacks,
            epochs=self.initial_steps // self.val_freq + self.steps // self.val_freq,
            initial_epoch=self.initial_steps // self.val_freq,
            steps_per_epoch=self.val_freq,
            validation_data=tf_val_data,
            validation_freq=1,
        )
        history_filepath = os.path.join(self.local_path, "history.json")
        with open(history_filepath, "w") as history_file:
            json.dump(history.history, history_file, indent=2)
            logger.info("Written training history to %s", history_filepath)

        if self.stream_to_wandb:
            wandb.finish()

        # set checkpoint name
        self.pipeline.checkpoint_name = self.checkpoint_name
        # reset eval model
        self.pipeline._eval_model = None

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_tf_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "dtype_float"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_tf_float_dtype[config[name]]

        def preprocess_tf_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "dtype_int"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_tf_int_dtype[config[name]]

        def preprocess_optimizer(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "optimizer"
            if name in config and config[name] is not None:
                config[name] = load_tf_optimizer_from_config(config[name])

        return super().config_preprocessors() + [
            preprocess_tf_float_dtype,
            preprocess_tf_int_dtype,
            preprocess_optimizer,
        ]
