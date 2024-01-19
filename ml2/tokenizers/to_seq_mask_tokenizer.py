"""Tokenizers that encode to a sequence encoding with mask of various dimensions"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Tuple, TypeVar

import numpy as np
import tensorflow as tf
import torch

from ..dtypes import DType
from ..utils.np_utils import np_float_dtype_to_str, str_to_np_float_dtype
from ..utils.pt_utils import pt_float_dtype_to_str, str_to_pt_float_dtype
from ..utils.tf_utils import str_to_tf_float_dtype, tf_float_dtype_to_str
from .to_seq_tokenizer import SeqEncoding, ToSeqTokenizer
from .tokenizer import TFEncoding

T = TypeVar("T", bound=DType)


class ToSeqMaskTokenizer(ToSeqTokenizer[T], Generic[T]):
    def __init__(
        self,
        mask_shape: Tuple,
        np_float_dtype: type = np.float32,
        pt_float_dtype: torch.dtype = torch.float32,
        tf_float_dtype: tf.DType = tf.float32,
        **kwargs,
    ):
        self.mask_shape = mask_shape
        self.np_float_dtype = np_float_dtype
        self.pt_float_dtype = pt_float_dtype
        self.tf_float_dtype = tf_float_dtype

        super().__init__(**kwargs)

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_np_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "np_float_dtype"
            if name in config and isinstance(config[name], type):
                config[name] = np_float_dtype_to_str[config[name]]

        def postprocess_pt_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "pt_float_dtype"
            if name in config and isinstance(config[name], torch.dtype):
                config[name] = pt_float_dtype_to_str[config[name]]

        def postprocess_tf_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "tf_float_dtype"
            if name in config and isinstance(config[name], tf.DType):
                config[name] = tf_float_dtype_to_str[config[name]]

        return [
            postprocess_np_float_dtype,
            postprocess_pt_float_dtype,
            postprocess_tf_float_dtype,
        ] + super().config_postprocessors()

    @property
    def tf_signature(self) -> Tuple[tf.TypeSpec, tf.TypeSpec]:
        seq_spec = tf.TensorSpec(shape=(self.pad,), dtype=self.tf_int_dtype)
        mask_spec = tf.TensorSpec(shape=self.mask_shape, dtype=self.tf_float_dtype)
        return (seq_spec, mask_spec)

    def encode_tf(self, data: T, **kwargs) -> TFEncoding:
        encoding = self.encode(data, **kwargs)
        ids_tensor = tf.constant(encoding.ids, dtype=self.tf_int_dtype)
        pad_mask_tensor = tf.constant(encoding.pad_mask, dtype=self.tf_float_dtype)
        return (ids_tensor, pad_mask_tensor)

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_np_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "np_float_dtype"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_np_float_dtype[config[name]]

        def preprocess_pt_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "pt_float_dtype"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_pt_float_dtype[config[name]]

        def preprocess_tf_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "tf_float_dtype"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_tf_float_dtype[config[name]]

        return super().config_preprocessors() + [
            preprocess_np_float_dtype,
            preprocess_pt_float_dtype,
            preprocess_tf_float_dtype,
        ]


class Seq2DMaskEncoding(SeqEncoding):
    def __init__(
        self,
        tokens: List[str],
        pad_tokens: List[str],
        ids: List[int],
        mask: List[List[float]],
        pad_mask: List[List[float]],
    ) -> None:
        self.mask = mask
        self.pad_mask = pad_mask
        super().__init__(
            tokens=tokens,
            pad_tokens=pad_tokens,
            ids=ids,
        )


class ToSeq2DMaskTokenizer(ToSeqMaskTokenizer[T], Generic[T]):
    def __init__(self, mask_shape: Tuple[int, int], **kwargs):
        super().__init__(mask_shape=mask_shape, **kwargs)

    @abstractmethod
    def encode_mask(self, data: T, **kwargs) -> List[List[float]]:
        raise NotImplementedError()

    @abstractmethod
    def pad_mask(self, mask: List[List[float]]) -> List[List[float]]:
        raise NotImplementedError()


class Seq3DMaskEncoding(SeqEncoding):
    def __init__(
        self,
        tokens: List[str],
        pad_tokens: List[str],
        ids: List[int],
        masks: List[List[List[float]]],
        pad_masks: List[List[List[float]]],
    ) -> None:
        self.masks = masks
        self.pad_masks = pad_masks
        super().__init__(
            tokens=tokens,
            pad_tokens=pad_tokens,
            ids=ids,
        )


class ToSeq3DMaskTokenizer(ToSeqMaskTokenizer[T], Generic[T]):
    def __init__(self, mask_shape: Tuple[int, int, int], **kwargs):
        super().__init__(mask_shape=mask_shape, **kwargs)

    @abstractmethod
    def encode_mask(self, data: T, **kwargs) -> List[List[List[float]]]:
        raise NotImplementedError()

    @abstractmethod
    def pad_mask(self, mask: List[List[List[float]]]) -> List[List[List[float]]]:
        raise NotImplementedError()


class Seq4DMaskEncoding(SeqEncoding):
    def __init__(
        self,
        tokens: List[str],
        pad_tokens: List[str],
        ids: List[int],
        masks: List[List[List[List[float]]]],
        pad_masks: List[List[List[List[float]]]],
    ) -> None:
        self.masks = masks
        self.pad_masks = pad_masks
        super().__init__(
            tokens=tokens,
            pad_tokens=pad_tokens,
            ids=ids,
        )


class ToSeq4DMaskTokenizer(ToSeqMaskTokenizer[T], Generic[T]):
    def __init__(self, mask_shape: Tuple[int, int, int, int], **kwargs):
        super().__init__(mask_shape=mask_shape, **kwargs)

    @abstractmethod
    def encode_mask(self, data: T, **kwargs) -> List[List[List[List[float]]]]:
        raise NotImplementedError()

    @abstractmethod
    def pad_mask(self, mask: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
        raise NotImplementedError()
