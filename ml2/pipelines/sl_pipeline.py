"""Supervised learning pipeline"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, Generic, List, TypeVar

from ..datasets import Dataset
from ..dtypes import DType, Supervised
from ..registry import register_type
from ..tokenizers import Tokenizer
from .callbacks.callback import Callback
from .metrics import Metric
from .pipeline import EvalTask, Pipeline
from .samples import EvalLabeledSample, EvalSample

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class SLPipeline(Pipeline[I], Generic[I, T]):
    def __init__(
        self,
        input_tokenizer: Tokenizer[I] = None,
        target_tokenizer: Tokenizer[T] = None,
        vocab_dataset: Dataset[Supervised[I, T]] = None,
        **kwargs,
    ) -> None:
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer

        super().__init__(**kwargs)

        if vocab_dataset is not None:
            self.build_vocabulary(vocab_dataset)

    @property
    def input_vocab_size(self) -> int:
        return self.input_tokenizer.vocabulary.size()

    @property
    def target_vocab_size(self) -> int:
        return self.target_tokenizer.vocabulary.size()

    @abstractmethod
    def __call__(self, x: I, **kwargs) -> T:
        raise NotImplementedError()

    def build_vocabulary(self, dataset: Dataset[Supervised[I, T]]) -> None:
        def input_generator():
            for x in dataset.generator():
                yield x.input

        def target_generator():
            for x in dataset.generator():
                yield x.target

        logger.info(f"Building input vocabulary with dataset {dataset.name}")
        self.input_tokenizer.build_vocabulary(input_generator())
        logger.info(f"Building target vocabulary with dataset {dataset.name}")
        self.target_tokenizer.build_vocabulary(target_generator())

    @abstractmethod
    def eval(
        self,
        dataset: Dataset[I],
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        training: bool = False,
        **kwargs,
    ) -> Generator[EvalSample[I, T], None, None]:
        raise NotImplementedError()

    @abstractmethod
    def eval_sample(self, x: I, **kwargs) -> EvalSample[I, T]:
        raise NotImplementedError()

    @abstractmethod
    def eval_supervised(
        self,
        dataset: Dataset[Supervised[I, T]],
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        training: bool = False,
        **kwargs,
    ) -> Generator[EvalLabeledSample[I, T], None, None]:
        raise NotImplementedError()

    @abstractmethod
    def eval_supervised_sample(self, x: Supervised[I, T], **kwargs) -> EvalLabeledSample[I, T]:
        raise NotImplementedError()

    def save(
        self,
        add_to_wandb: bool = False,
        overwrite_bucket: bool = False,
        overwrite_local: bool = False,
        recurse: bool = False,
        upload: bool = False,
    ) -> None:
        super().save(
            upload=upload,
            overwrite_local=overwrite_local,
            overwrite_bucket=overwrite_bucket,
            add_to_wandb=add_to_wandb,
        )

        if recurse:
            self.input_tokenizer.save(
                add_to_wandb=add_to_wandb,
                overwrite_bucket=overwrite_bucket,
                overwrite_local=overwrite_local,
                recurse=True,
                upload=upload,
            )

            self.target_tokenizer.save(
                add_to_wandb=add_to_wandb,
                overwrite_bucket=overwrite_bucket,
                overwrite_local=overwrite_local,
                recurse=True,
                upload=upload,
            )

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_tokenizer_names(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            if "input_tokenizer" in config and isinstance(config["input_tokenizer"], dict):
                config["input_tokenizer"]["default_name"] = "input-tokenizer"

            if "target_tokenizer" in config and isinstance(config["target_tokenizer"], dict):
                config["target_tokenizer"]["default_name"] = "target-tokenizer"

        return [preprocess_tokenizer_names] + super().config_preprocessors()

    @staticmethod
    @abstractmethod
    def default_supervised_metric() -> Metric:
        raise NotImplementedError()


@register_type
class SupervisedEvalTask(EvalTask, Generic[I, T]):
    @property
    def dataset(self) -> Dataset[Supervised[I, T]]:
        return super().dataset

    @property
    def metric(self) -> Metric:
        if self._metric is None:
            if self.metric_config is None:
                self._metric = self.pipeline.default_supervised_metric()
            else:
                from .metrics import load_metric_from_config

                self._metric = load_metric_from_config(self.metric_config)
        return self._metric

    @property
    def pipeline(self) -> SLPipeline[I, T]:
        return super().pipeline

    def run(self):
        for _ in self.pipeline.eval_supervised(
            dataset=self.dataset,
            batch_size=self.batch_size,
            metric=self.metric,
            callbacks=self.callbacks,
            return_errors=True,
        ):
            pass
