"""TensorFlow hierarchical Transformer pipeline"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, TypeVar

import tensorflow as tf

from ..configurable import Configurable
from ..dtypes import DType
from ..models import tf_hierarchical_transformer
from ..registry import register_type
from ..utils.tf_utils import (
    str_to_tf_float_dtype,
    str_to_tf_int_dtype,
    tf_float_dtype_to_str,
    tf_int_dtype_to_str,
)
from .tf_transformer_pipeline import TFTransformerPipeline

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
@register_type
class TFHierTransformerConfig(Configurable):
    alpha: float = 0.5
    beam_size: int = 2
    d_embed: int = 256
    d_embed_dec: int = None
    d_embed_enc: int = None
    d_ff: int = 1024
    d_ff_dec: int = None
    d_ff_enc: int = None
    d_ff_enc_d0: int = None
    d_ff_enc_d1: int = None
    dropout: float = 0.0
    dropout_dec: float = None
    dropout_enc: float = None
    dtype_float: tf.DType = tf.float32
    dtype_int: tf.DType = tf.int32
    ff_activation: str = "relu"
    ff_activation_dec: str = None
    ff_activation_enc: str = None
    ff_activation_enc_d0: str = None
    ff_activation_enc_d1: str = None
    num_heads: int = 4
    num_heads_dec: int = None
    num_heads_enc: int = None
    num_heads_enc_d0: int = None
    num_heads_enc_d1: int = None
    num_layers: int = 8
    num_layers_dec: int = None
    num_layers_enc: int = None
    num_layers_enc_d0: int = None
    num_layers_enc_d1: int = None

    def __post_init__(self):
        if self.d_embed is not None:
            if self.d_embed_enc is None:
                self.d_embed_enc = self.d_embed
            if self.d_embed_dec is None:
                self.d_embed_dec = self.d_embed
        if self.d_ff is not None:
            if self.d_ff_enc is None:
                self.d_ff_enc = self.d_ff
            if self.d_ff_dec is None:
                self.d_ff_dec = self.d_ff
        if self.d_ff_enc is not None:
            if self.d_ff_enc_d0 is None:
                self.d_ff_enc_d0 = self.d_ff_enc
            if self.d_ff_enc_d1 is None:
                self.d_ff_enc_d1 = self.d_ff_enc
        if self.dropout is not None:
            if self.dropout_enc is None:
                self.dropout_enc = self.dropout
            if self.dropout_dec is None:
                self.dropout_dec = self.dropout
        if self.ff_activation is not None:
            if self.ff_activation_enc is None:
                self.ff_activation_enc = self.ff_activation
            if self.ff_activation_dec is None:
                self.ff_activation_dec = self.ff_activation
        if self.ff_activation_enc is not None:
            if self.ff_activation_enc_d0 is None:
                self.ff_activation_enc_d0 = self.ff_activation_enc
            if self.ff_activation_enc_d1 is None:
                self.ff_activation_enc_d1 = self.ff_activation_enc
        if self.num_heads is not None:
            if self.num_heads_enc is None:
                self.num_heads_enc = self.num_heads
            if self.num_heads_dec is None:
                self.num_heads_dec = self.num_heads
        if self.num_heads_enc is not None:
            if self.num_heads_enc_d0 is None:
                self.num_heads_enc_d0 = self.num_heads_enc
            if self.num_heads_enc_d1 is None:
                self.num_heads_enc_d1 = self.num_heads_enc
        if self.num_layers is not None:
            if self.num_layers_enc is None:
                self.num_layers_enc = self.num_layers
            if self.num_layers_dec is None:
                self.num_layers_dec = self.num_layers
        if self.num_layers_enc is not None:
            if self.num_layers_enc_d0 is None:
                self.num_layers_enc_d0 = self.num_layers_enc
            if self.num_layers_enc_d1 is None:
                self.num_layers_enc_d1 = self.num_layers_enc
        if self.d_embed_enc % self.num_heads_enc_d0 != 0:
            raise ValueError(
                f"Encoder embedding dimension {self.d_embed_enc} is not divisible by the number of d0 encoder attention heads {self.num_heads_enc_d0}"
            )
        if self.d_embed_enc % self.num_heads_enc_d1 != 0:
            raise ValueError(
                f"Encoder embedding dimension {self.d_embed_enc} is not divisible by the number of d1 encoder attention heads {self.num_heads_enc_d1}"
            )
        if self.d_embed_dec % self.num_heads_dec != 0:
            raise ValueError(
                f"Decoder embedding dimension {self.d_embed_dec} is not divisible by the number of decoder attention heads {self.num_heads_dec}"
            )

    def asdict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in [
                "d_embed",
                "d_ff",
                "d_ff_enc",
                "dropout",
                "ff_activation",
                "ff_activation_enc",
                "num_heads",
                "num_heads_enc",
                "num_layers",
                "num_layers_enc",
            ]
        }

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_abbreviations(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            for abb in [
                "d_embed",
                "d_ff",
                "d_ff_enc",
                "dropout",
                "ff_activation",
                "ff_activation_enc",
                "num_heads",
                "num_heads_enc",
                "num_layers",
                "num_layers_enc",
            ]:
                config.pop(abb, None)
                annotations.pop(abb, None)

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
            postprocess_abbreviations,
            postprocess_tf_float_dtype,
            postprocess_tf_int_dtype,
        ] + super().config_postprocessors()

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

        return [
            preprocess_tf_float_dtype,
            preprocess_tf_int_dtype,
        ] + super().config_preprocessors()


@register_type
class TFHierTransformerPipeline(TFTransformerPipeline[I, T]):
    def __init__(
        self,
        max_local_length: int,
        max_local_num: int,
        model_config: TFHierTransformerConfig,
        fix_local_embed: bool = False,
        **kwargs,
    ) -> None:
        self.max_local_length = max_local_length
        self.max_local_num = max_local_num
        self.fix_local_embed = fix_local_embed
        super().__init__(model_config=model_config, **kwargs)

    def init_model(self, training: bool = False, attn_weights: bool = False, **kwargs):
        config = asdict(self.model_config)
        config["fix_d1_embed"] = self.fix_local_embed
        config["input_vocab_size"] = self.input_vocab_size
        config["input_eos_id"] = self.input_eos_id
        config["input_pad_id"] = self.input_pad_id
        config["target_vocab_size"] = self.target_vocab_size
        config["target_start_id"] = self.target_start_id
        config["target_eos_id"] = self.target_eos_id
        config["target_pad_id"] = self.target_pad_id
        config["input_length"] = (self.max_local_num, self.max_local_length)
        config["max_decode_length"] = self.max_target_length
        config["num_replica"] = getattr(self, "num_replica", None)

        return tf_hierarchical_transformer.create_model(
            config,
            training=training,
            custom_pos_enc=self.custom_pos_enc,
            attn_weights=attn_weights,
        )

    def eval_attn_weights(self, x: I, training: bool = False):
        raise NotImplementedError()

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_max_input_length(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            if "max_input_length" not in config:
                config["max_input_length"] = config["max_local_num"] * config["max_local_length"]

        return super().config_preprocessors() + [preprocess_max_input_length]
