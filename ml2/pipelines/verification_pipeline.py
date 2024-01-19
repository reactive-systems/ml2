""""Verification pipeline"""

import copy
import logging
import time
from typing import Any, Generator, Generic, List, Optional, Tuple, TypeVar

from tqdm import tqdm

from ..datasets import Dataset
from ..dtypes import DType, Supervised, ValidationResult
from ..registry import register_type
from ..verifier import Verifier
from .callbacks.callback import Callback
from .metrics import (
    Acc,
    AccPerSeq,
    Counter,
    EvalSupervisedErrCounter,
    Metric,
    MetricGroup,
    NullMetric,
    SemAcc,
    VerificationErrCounter,
    VerificationSupervisedErrCounter,
)
from .samples import EvalLabeledSample, EvalSample, VerifiedLabeledSample, VerifiedSample
from .sl_pipeline import SLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)
V = TypeVar("V", bound=ValidationResult)


@register_type
class VerificationPipeline(SLPipeline[I, T], Generic[I, T]):
    def __init__(self, pipeline: SLPipeline[I, T], verifier: Verifier, **kwargs) -> None:
        self.pipeline = pipeline
        self.verifier = verifier
        super().__init__(**kwargs)

    def __call__(self, x: I) -> Any:
        return self.eval_sample(x).verification

    def eval_sample(self, x: I, **kwargs) -> VerifiedSample[I, T, V]:
        return self.verify_sample(self.pipeline.eval_sample(x))

    def eval(
        self,
        dataset: Dataset[I],
        batch_size: int = 32,
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        **kwargs
    ) -> Generator[VerifiedSample[I, T, V], None, None]:
        if metric is None:
            metric = NullMetric()
        if callbacks is None:
            callbacks = []

        pbar = tqdm(desc="Verified samples", unit="sample")
        pbar.set_postfix(metric.compute_dict())

        for sample in self.pipeline.eval(
            dataset=dataset, batch_size=batch_size, return_errors=True, **kwargs
        ):
            verified_sample = self.verify_sample(sample)
            metric.add(verified_sample)
            [callback.add(verified_sample) for callback in callbacks]
            pbar.update()
            pbar.set_postfix(metric.compute_dict())

            if not return_errors and (
                verified_sample.verification is None
                or verified_sample.verification_err is not None
            ):
                continue

            yield verified_sample

    def eval_supervised_sample(
        self, x: Supervised[I, T], **kwargs
    ) -> VerifiedLabeledSample[I, T, V]:
        return self.verify_supervised_sample(self.pipeline.eval_supervised_sample(x))

    def eval_supervised(
        self,
        dataset: Dataset[Supervised[I, T]],
        batch_size: int = 32,
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        **kwargs
    ) -> Generator[VerifiedLabeledSample[I, T, V], None, None]:
        if metric is None:
            metric = NullMetric()
        if callbacks is None:
            callbacks = []

        pbar = tqdm(desc="Verified samples", unit="sample")
        pbar.set_postfix(metric.compute_dict())

        for sample in self.pipeline.eval_supervised(
            dataset=dataset, batch_size=batch_size, return_errors=True, **kwargs
        ):
            verified_sample = self.verify_supervised_sample(sample, **kwargs)
            metric.add(verified_sample)
            [callback.add(verified_sample) for callback in callbacks]
            pbar.update()
            pbar.set_postfix(metric.compute_dict())

            if not return_errors and (
                verified_sample.verification is None
                or verified_sample.verification_err is not None
            ):
                continue

            yield verified_sample

    def save(
        self,
        add_to_wandb: bool = False,
        overwrite_bucket: bool = False,
        overwrite_local: bool = False,
        recurse: bool = False,
        upload: bool = False,
    ):
        super().save(
            add_to_wandb=add_to_wandb,
            overwrite_bucket=overwrite_bucket,
            overwrite_local=overwrite_local,
            upload=upload,
        )

        if recurse:
            self.pipeline.save(
                add_to_wandb=add_to_wandb,
                overwrite_bucket=overwrite_bucket,
                overwrite_local=overwrite_local,
                recurse=recurse,
                upload=upload,
            )

    def verify_sample(self, sample: EvalSample[I, T]) -> VerifiedSample[I, T, V]:
        start = time.time()
        try:
            verification = (
                self.verifier.verify(sample.inp, sample.pred) if sample.pred is not None else None
            )
            verification_err = None
            end = time.time()
        except Exception as err:
            verification = None
            verification_err = str(err)
            end = time.time()
        return VerifiedSample(
            inp=sample.inp,
            inp_enc=sample.inp_enc,
            inp_enc_err=sample.inp_enc_err,
            id=sample.id,
            pred=sample.pred,
            pred_enc=sample.pred_enc,
            pred_dec_err=sample.pred_dec_err,
            verification=verification,
            verification_err=verification_err,
            time=(sample.time + (end - start)) if sample.time is not None else None,
            verification_time=end - start,
        )

    def verify_supervised_sample(
        self, sample: EvalLabeledSample[I, T], **kwargs
    ) -> VerifiedLabeledSample[I, T, V]:
        start = time.time()
        try:
            verification: Optional[ValidationResult] = (
                self.verifier.verify(sample.inp, sample.pred, **kwargs)
                if sample.pred is not None
                else None
            )
            verification_err = None
            end = time.time()
        except Exception as err:
            verification = None
            verification_err = str(err)
            end = time.time()
        return VerifiedLabeledSample(
            inp=sample.inp,
            inp_enc=sample.inp_enc,
            inp_enc_err=sample.inp_enc_err,
            id=sample.id,
            name=sample.name,
            tar=sample.tar,
            tar_enc=sample.tar_enc,
            tar_enc_err=sample.tar_enc_err,
            pred=sample.pred,
            pred_enc=sample.pred_enc,
            pred_dec_err=sample.pred_dec_err,
            verification=verification,
            verification_err=verification_err,
            time=(sample.time + (end - start)) if sample.time is not None else None,
            verification_time=end - start,
        )

    @staticmethod
    def default_metric() -> Metric:
        return MetricGroup([Counter(), SemAcc(), VerificationErrCounter()])

    @staticmethod
    def default_supervised_metric() -> Metric:
        return MetricGroup(
            [
                Acc(pad_same_length=True),
                AccPerSeq(pad_same_length=True),
                Counter(),
                EvalSupervisedErrCounter(),
                SemAcc(),
                VerificationSupervisedErrCounter(),
            ]
        )

    @staticmethod
    def expand_eval_config(config: dict) -> List[Tuple[dict, dict]]:
        expanded_configs = []
        pipeline_config = config["pipeline"]
        if isinstance(pipeline_config, dict) and "pipeline" in pipeline_config:
            from ..loading import get_artifact_type

            pipeline_type = get_artifact_type(pipeline_config["pipeline"])
            for delimiters, expanded_pipeline_config in pipeline_type.expand_eval_config(
                pipeline_config
            ):
                expanded_config = copy.deepcopy(config)
                expanded_config["pipeline"] = expanded_pipeline_config
                expanded_delimiters = {"pipeline": delimiters}
                expanded_configs.append((expanded_delimiters, expanded_config))
        else:
            expanded_configs.append(({}, config))

        return expanded_configs
