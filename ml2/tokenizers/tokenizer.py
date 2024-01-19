"""Abstract tokenizer class"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, Generic, List, Tuple, Type, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from ..artifact import Artifact
from ..configurable import Configurable
from ..datasets import Dataset
from ..dtypes import DType
from ..registry import register_type
from ..utils.np_utils import np_int_dtype_to_str, str_to_np_int_dtype
from ..utils.pt_utils import pt_int_dtype_to_str, str_to_pt_int_dtype
from ..utils.tf_utils import str_to_tf_int_dtype, tf_int_dtype_to_str
from .vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_TOKEN = "<s>"
EOS_TOKEN = "<e>"
PAD_TOKEN = "<p>"


T = TypeVar("T", bound=DType)

NPEncoding = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
PTEncoding = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
TFEncoding = Union[tf.Tensor, Dict[str, tf.Tensor], Tuple[tf.Tensor, ...]]


class Encoding(object):
    def __init__(self, tokens: List[str], ids: List[int]) -> None:
        self.tokens = tokens
        self.ids = ids


class TokenizationException(Exception):
    pass


@register_type
class Tokenizer(Artifact, Configurable, Generic[T]):
    def __init__(
        self,
        dtype: Type[T],
        vocabulary: Vocabulary = None,
        np_int_dtype: type = np.int32,
        pt_int_dtype: torch.dtype = torch.int32,
        tf_int_dtype: tf.DType = tf.int32,
        name: str = "tokenizer",
        **kwargs,
    ):
        self.dtype = dtype
        self.np_int_dtype = np_int_dtype
        self.pt_int_dtype = pt_int_dtype
        self.tf_int_dtype = tf_int_dtype

        self._vocabulary = vocabulary

        super().__init__(name=name, **kwargs)

    @property
    @abstractmethod
    def tf_signature(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec], Tuple[tf.TypeSpec, ...]]:
        raise NotImplementedError()

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            raise Exception("No vocabulary given or build")
        return self._vocabulary

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_np_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "np_int_dtype"
            if name in config and isinstance(config[name], type):
                config[name] = np_int_dtype_to_str[config[name]]

        def postprocess_pt_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "pt_int_dtype"
            if name in config and isinstance(config[name], torch.dtype):
                config[name] = pt_int_dtype_to_str[config[name]]

        def postprocess_tf_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "tf_int_dtype"
            if name in config and isinstance(config[name], tf.DType):
                config[name] = tf_int_dtype_to_str[config[name]]

        return [
            postprocess_np_int_dtype,
            postprocess_pt_int_dtype,
            postprocess_tf_int_dtype,
        ] + super().config_postprocessors()

    @abstractmethod
    def encode(self, data: T, **kwargs) -> Encoding:
        raise NotImplementedError()

    def encode_dataset(self, dataset: Dataset[T], tensor_type: str = None, **kwargs):
        if tensor_type == "tf":
            return self.encode_dataset_tf(dataset)
        else:
            raise NotImplementedError()

    @abstractmethod
    def encode_np(self, data: T, **kwargs) -> NPEncoding:
        raise NotImplementedError()

    @abstractmethod
    def encode_pt(self, data: T, **kwargs) -> PTEncoding:
        raise NotImplementedError()

    @abstractmethod
    def encode_tf(self, data: T, **kwargs) -> TFEncoding:
        raise NotImplementedError()

    def encode_dataset_tf(
        self, dataset: Dataset[T], errors: Dict[str, int] = None, **kwargs
    ) -> tf.data.Dataset:
        def generator():
            for sample in dataset.generator():
                try:
                    yield self.encode_tf(sample, **kwargs)
                except Exception as error:
                    if errors is not None:
                        error = str(error)
                        errors[error] = errors.get(error, 0) + 1
                    continue

        return tf.data.Dataset.from_generator(generator, output_signature=self.tf_signature)

    @abstractmethod
    def decode(self, ids: Any, **kwargs) -> T:
        raise NotImplementedError()

    @abstractmethod
    def decode_np(self, encoding: NPEncoding, **kwargs) -> T:
        raise NotImplementedError()

    @abstractmethod
    def decode_pt(self, encoding: PTEncoding, **kwargs) -> T:
        raise NotImplementedError()

    @abstractmethod
    def decode_tf(self, encoding: TFEncoding, **kwargs) -> T:
        raise NotImplementedError()

    def load_vocabulary(self, path: str) -> bool:
        filename = self.vocabulary_filename()
        self._vocabulary = Vocabulary.from_file(path, filename)
        return self.vocabulary is not None

    @abstractmethod
    def build_vocabulary(self, generator: Generator[T, None, None], **kwargs) -> None:
        raise NotImplementedError()

    def vocabulary_to_file(self, path: str) -> None:
        self.vocabulary.to_file(path, self.vocabulary_filename())

    def vocabulary_filename(self) -> str:
        return self.dtype.__name__.lower() + "-vocab"

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
            upload=upload,
        )

        if recurse:
            self.vocabulary.save(
                add_to_wandb=add_to_wandb,
                overwrite_bucket=overwrite_bucket,
                overwrite_local=overwrite_local,
                upload=upload,
            )

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_np_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "np_int_dtype"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_np_int_dtype[config[name]]

        def preprocess_pt_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "pt_int_dtype"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_pt_int_dtype[config[name]]

        def preprocess_tf_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "tf_int_dtype"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_tf_int_dtype[config[name]]

        return super().config_preprocessors() + [
            preprocess_np_int_dtype,
            preprocess_pt_int_dtype,
            preprocess_tf_int_dtype,
        ]
