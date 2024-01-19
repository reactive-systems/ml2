"""TensorFlow Transformer learning rate schedule"""

import tensorflow as tf

from ...registry import register_type


class TransformerScheduleArgs:
    def __init__(self, warmup_steps: int = 4000) -> None:
        self.warmup_steps = warmup_steps


@tf.keras.utils.register_keras_serializable(package="ml2")
@register_type
class TFTransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning Rate Schedule proposed by Vaswani et al. (2017) that corresponds to a linear increase
    during the warmup phase followed by a decrease proportional to the inverse of the square root of
    the step number"""

    def __init__(self, d_embed: int, warmup_steps: int = 4000) -> None:
        super().__init__()
        self.d_embed = tf.cast(d_embed, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        increasing_lr = step * (self.warmup_steps**-1.5)
        decreasing_lr = tf.math.rsqrt(step)
        return tf.math.rsqrt(self.d_embed) * tf.math.minimum(increasing_lr, decreasing_lr)

    def get_config(self) -> dict:
        return {
            "name": "TFTransformerLRSchedule",
            "d_embed": int(self.d_embed),
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "TFTransformerLRSchedule":
        if "d_embed" not in config:
            raise Exception("Embedding dimension not specified in learning rate schedule configs")
        return cls(d_embed=config["d_embed"], warmup_steps=config.get("warmup_steps", 4000))
