"""Abstract pipeline class"""

import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Generator, Generic, List, Tuple, TypeVar

from ..artifact import Artifact
from ..datasets import Dataset
from ..dtypes import DType
from ..registry import register_type
from .callbacks import Callback
from .loggers import CSVToDatasetLogger
from .metrics import Metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=DType)


@register_type
class Pipeline(Artifact, Generic[T]):
    WANDB_TYPE = "pipe"

    def __init__(self, name: str = "pipe", group: str = None, **kwargs) -> None:
        self.group = group

        super().__init__(name=name, **kwargs)

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.local_path, "eval")

    @property
    def temp_dir(self) -> str:
        temp_path = os.path.join(self.local_path, "temp")
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)
        return temp_path

    @abstractmethod
    def __call__(self, x: T, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def eval(
        self,
        dataset: Dataset[T],
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        training: bool = False,
        **kwargs,
    ) -> Generator:
        raise NotImplementedError()

    @abstractmethod
    def eval_sample(self, x: T, **kwargs):
        raise NotImplementedError()

    # @abstractmethod
    # def eval_generator(self, generator, training: bool = False, **kwargs):
    #     raise NotImplementedError()

    # @abstractmethod
    # def eval_ray_queue(self, queue, **kwargs):
    #     raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def default_metric() -> Metric:
        raise NotImplementedError()

    @staticmethod
    def expand_eval_config(config: dict) -> List[Tuple[dict, dict]]:
        return [({}, config)]


@register_type
class EvalTask(Artifact, Generic[T]):
    def __init__(
        self,
        pipeline_config,
        dataset_config,
        batch_size: int,
        name: str,
        metric_config: dict = None,
        stream_to_wandb: bool = False,
        callbacks_config: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        if callbacks_config is None:
            callbacks_config = []
        self.pipeline_config = pipeline_config
        self.dataset_config = dataset_config
        self.metric_config = metric_config
        self.callbacks_config = callbacks_config
        self.batch_size = batch_size
        self._dataset = None
        self._metric = None
        self._callbacks = None
        self._pipeline = None

        self.stream_to_wandb = stream_to_wandb
        super().__init__(name=name, **kwargs)

    @property
    def dataset(self) -> Dataset[T]:
        from ..loading import load_artifact

        if self._dataset is None:
            self._dataset = load_artifact(self.dataset_config)
        return self._dataset

    @property
    def metric(self) -> Metric:
        if self._metric is None:
            if self.metric_config is None:
                self._metric = self.pipeline.default_metric()
            else:
                from .metrics import load_metric_from_config

                self._metric = load_metric_from_config(self.metric_config)
        return self._metric

    @property
    def callbacks(self) -> List[Callback]:
        if self._callbacks is None:
            self._callbacks = []
            self._callbacks.append(
                CSVToDatasetLogger(name=self.name + "/csv_logger", project=self.project)
            )

        return self._callbacks

    @property
    def pipeline(self) -> Pipeline[T]:
        from ..loading import load_artifact

        if self._pipeline is None:
            self._pipeline = load_artifact(self.pipeline_config)
        return self._pipeline

    def run(self):
        for _ in self.pipeline.eval(
            dataset=self.dataset,
            batch_size=self.batch_size,
            metric=self.metric,
            callbacks=self.callbacks,
            return_errors=True,
        ):
            pass

    def save_to_path(self, path: str) -> None:
        # Assume always called at end of task
        if not os.path.exists(self.pipeline.local_path):
            self.pipeline.save()
        self.metric.save_to_path(path=path)
        [callback.save() for callback in self.callbacks]
        super().save_to_path(path)

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_renaming(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "dataset" in config and "dataset_config" not in config:
                config["dataset_config"] = config.pop("dataset")
            if "pipeline" in config and "pipeline_config" not in config:
                config["pipeline_config"] = config.pop("pipeline")
            if "metric" in config:
                metric = config.pop("metric")
                if "metric_config" not in config:
                    config["metric_config"] = metric

        def preprocess_pipeline_name(config: Dict[str, Any], annotations: Dict[str, type]):
            if (
                "pipeline_config" in config
                and isinstance(config["pipeline_config"], dict)
                and "name" not in config["pipeline_config"]
            ):
                config["pipeline_config"]["name"] = config["name"] + "/test-pipeline"

        return [
            preprocess_renaming,
            preprocess_pipeline_name,
        ] + super().config_preprocessors()
