"""Keras callbacks"""

import logging

import tensorflow as tf

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WandbCallback(tf.keras.callbacks.Callback):
    """WandB callback inspired by WandB's Keras integration: https://github.com/wandb/wandb/blob/51a7e224b477fcc044ed607e53153d15e2583809/wandb/integration/keras/callbacks/metrics_logger.py"""

    def __init__(
        self,
        log_freq: int,
        initial_global_step: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if wandb.run is None:
            raise Exception("wandb not initialized prior to WandbCallback construction")

        self.log_freq = log_freq
        self.global_step = initial_global_step

        # define x-axis as global_step
        wandb.define_metric("train/global_step")
        # log train and evaluation metrics against global_step
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")

    def _get_lr(self) -> float | None:
        if isinstance(
            self.model.optimizer.learning_rate,
            (tf.Variable, tf.Tensor),
        ) or (
            hasattr(self.model.optimizer.learning_rate, "shape")
            and self.model.optimizer.learning_rate.shape == ()
        ):
            return float(self.model.optimizer.learning_rate.numpy().item())
        try:
            return float(self.model.optimizer.learning_rate(step=self.global_step).numpy().item())
        except Exception as err:
            logger.warning(f"When logging the learning rate an exception was raised: {err}")
            return None

    def on_test_end(self, logs: dict | None = None) -> None:
        logs = dict() if logs is None else {f"eval/{k}": v for k, v in logs.items()}
        wandb.log(logs)

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.global_step += 1
        if batch % self.log_freq == 0:
            logs = {f"train/{k}": v for k, v in logs.items()} if logs else {}
            logs["train/global_step"] = self.global_step
            if (lr := self._get_lr()) is not None:
                logs["train/learning_rate"] = lr
            wandb.log(logs)
