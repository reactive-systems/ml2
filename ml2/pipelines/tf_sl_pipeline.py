"""TensorFlow supervised learning pipeline"""

import json
import logging
import os
from typing import Dict, Generator, Generic, TypeVar

import tensorflow as tf
from tensorflow import keras

from ..datasets import Dataset
from ..dtypes import DType, Supervised
from ..registry import register_type
from .samples import EncodedSample, LabeledSample
from .sl_pipeline import SLPipeline
from .tf_pipeline import TFPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)


class TokenizationErrorCallback(keras.callbacks.Callback):
    def __init__(
        self, name: str, errors: Dict[str, int], log_dir: str, filename: str = None
    ) -> None:
        super().__init__()
        self.name = name
        self.errors = errors
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.filename = filename if filename is not None else self.name + "-err"

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            return
        if any(self.errors):
            logger.info(f"When tokenizing {self.name} errors occured: %s", self.errors)
            errs_filepath = os.path.join(self.log_dir, self.filename)
            with open(errs_filepath, "w") as errs_file:
                json.dump(self.errors, errs_file)


@register_type
class TFSLPipeline(TFPipeline[I], SLPipeline[I, T], Generic[I, T]):
    def init_sample(self, x: I) -> EncodedSample:
        try:
            input_tensor = self.input_tokenizer.encode_tf(x)
            return EncodedSample(
                inp=x,
                inp_enc=input_tensor,
                id=None,
                name=x.name if hasattr(x, "name") else None,
            )
        except Exception as err:
            return EncodedSample(
                inp=x,
                id=None,
                inp_enc_err=str(err),
                name=x.name if hasattr(x, "name") else None,
            )

    def init_samples(
        self, dataset: Dataset[I], return_errors: bool = True
    ) -> Generator[EncodedSample, None, None]:
        logger.info(f"Tokenizing dataset {dataset.name}")

        for id, x in enumerate(dataset.generator()):
            try:
                input_tensor = self.input_tokenizer.encode_tf(x)
                sample = EncodedSample(
                    inp=x,
                    inp_enc=input_tensor,
                    id=id,
                    name=x.name if hasattr(x, "name") else None,
                )
            except Exception as err:
                if return_errors:
                    yield EncodedSample(
                        inp=x,
                        inp_enc_err=str(err),
                        id=id,
                        name=x.name if hasattr(x, "name") else None,
                    )
                continue

            yield sample

    def init_supervised_samples(
        self, dataset: Dataset[Supervised[I, T]], return_errors: bool = True
    ) -> Generator[LabeledSample, None, None]:
        logger.info(f"Tokenizing dataset {dataset.name}")

        for id, x in enumerate(dataset.generator()):
            try:
                input_tensor = self.input_tokenizer.encode_tf(x.input)
                sample = LabeledSample(
                    inp=x.input,
                    tar=x.target,
                    inp_enc=input_tensor,
                    id=id,
                    name=x.name if hasattr(x, "name") else None,
                )
            except Exception as err:
                if return_errors:
                    yield LabeledSample(
                        inp=x.input,
                        tar=x.target,
                        inp_enc_err=str(err),
                        id=id,
                        name=x.name if hasattr(x, "name") else None,
                    )
                continue

            try:
                target_tensor = self.target_tokenizer.encode_tf(x.target)
                sample.tar_enc = target_tensor
            except Exception as err:
                sample.tar_enc_err = str(err)
                if return_errors:
                    yield sample
                continue

            yield sample

    def convert_sl_dataset_to_tf(
        self, dataset: Dataset[Supervised[I, T]], return_error_callbacks: bool = False
    ):
        input_tokenizer_errors: Dict[str, int] = {}
        target_tokenizer_errors: Dict[str, int] = {}

        output_signature = (
            self.input_tokenizer.tf_signature,
            self.target_tokenizer.tf_signature,
        )

        def tf_generator():
            for sample in dataset.generator():
                try:
                    input_tensor = self.input_tokenizer.encode_tf(sample.input)
                except Exception as err:
                    err_str = str(err) if str(err) != "" else err.__class__.__name__
                    input_tokenizer_errors[err_str] = input_tokenizer_errors.get(err_str, 0) + 1
                    continue

                try:
                    target_tensor = self.target_tokenizer.encode_tf(sample.target)
                except Exception as err:
                    err_str = str(err) if str(err) != "" else err.__class__.__name__
                    target_tokenizer_errors[err_str] = target_tokenizer_errors.get(err_str, 0) + 1
                    continue

                yield input_tensor, target_tensor

        tf_dataset = tf.data.Dataset.from_generator(
            tf_generator, output_signature=output_signature
        )

        if return_error_callbacks:
            log_dir = os.path.join(self.local_path, "tokenizer-errors")
            log_dir = os.path.join(log_dir, dataset.name.replace("/", "-"))
            inp_errs_callback = TokenizationErrorCallback(
                name=f"{dataset.name} inputs",
                errors=input_tokenizer_errors,
                log_dir=log_dir,
                filename="input-errors",
            )
            tar_errs_callback = TokenizationErrorCallback(
                name=f"{dataset.name} targets",
                errors=target_tokenizer_errors,
                log_dir=log_dir,
                filename="target-errors",
            )
            return tf_dataset, [inp_errs_callback, tar_errs_callback]
        else:
            return tf_dataset
