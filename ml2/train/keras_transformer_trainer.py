"""Keras Transformer trainer"""

import logging

from keras.optimizers import Adam, Optimizer

from ..optim.tf_optim.tf_transformer_lr_schedule import TFTransformerLRSchedule
from ..pipelines import TFPipeline
from ..registry import register_type
from .keras_trainer import KerasTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class KerasTransformerTrainer(KerasTrainer):
    def __init__(
        self, pipeline: TFPipeline, optimizer: Optimizer = None, warmup_steps: int = 4000, **kwargs
    ):
        if optimizer is None:
            learning_rate = TFTransformerLRSchedule(
                pipeline.model_config["d_embed_enc"], warmup_steps
            )
            optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.warmup_steps = warmup_steps

        super().__init__(pipeline=pipeline, optimizer=optimizer, **kwargs)
