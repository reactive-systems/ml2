"""HuggingFace PyTorch text to text pipeline"""

import json
import logging
import os
import time
from typing import Dict, Generator, Generic, List, Type, TypeVar

from datasets import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, TrainerCallback

from ...configurable import Configurable
from ...datasets import Dataset
from ...dtypes import String, Supervised
from ...registry import register_type
from ..callbacks import Callback
from ..metrics import (
    Acc,
    AccPerSeq,
    Counter,
    EvalErrCounter,
    EvalSupervisedErrCounter,
    Metric,
    MetricGroup,
    NullMetric,
)
from ..model_pipeline import ModelPipeline
from ..samples import Beam, BeamSearchLabeledSample, BeamSearchSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


I = TypeVar("I", bound=String)
T = TypeVar("T", bound=String)


class TokenizationErrorCallback(TrainerCallback):

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

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch <= 1.0 and any(self.errors):
            logger.info(f"When tokenizing {self.name} errors occured: %s", self.errors)
            errs_filepath = os.path.join(self.log_dir, self.filename)
            with open(errs_filepath, "w") as errs_file:
                json.dump(self.errors, errs_file)


class HFModelConfig:
    pass


@register_type
class HFPTText2TextPipeline(ModelPipeline, Generic[I, T]):
    # TODO make model_config type Union[str, Config]
    def __init__(
        self,
        hf_input_tokenizer: str,
        hf_target_tokenizer: str,
        max_input_length: int,
        max_target_length: int,
        beam_size: int = 1,
        input_dtype: Type[String] = String,
        target_dtype: Type[String] = String,
        input_kwargs: dict = None,
        target_kwargs: dict = None,
        model_config: Configurable = None,
        checkpoint_name: str = None,
        hf_checkpoint_name: str = None,
        prompt: str = None,
        **kwargs,
    ):
        if model_config is None and checkpoint_name is None and hf_checkpoint_name is None:
            raise ValueError("Model config and checkpoint name can not be None at the same time")

        self.hf_input_tokenizer = hf_input_tokenizer
        self.hf_target_tokenizer = hf_target_tokenizer
        self.input_tokenizer = AutoTokenizer.from_pretrained(hf_input_tokenizer)
        self.target_tokenizer = AutoTokenizer.from_pretrained(hf_target_tokenizer)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.beam_size = beam_size
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype
        self.input_kwargs = input_kwargs if input_kwargs is not None else {}
        self.target_kwargs = target_kwargs if target_kwargs is not None else {}
        self.hf_checkpoint_name = hf_checkpoint_name
        self.prompt = prompt if prompt is not None else ""

        self._train_model = None
        self._eval_model = None

        super().__init__(model_config=model_config, checkpoint_name=checkpoint_name, **kwargs)

    @property
    def eval_model(self):
        if not self._eval_model:
            self._eval_model = self.init_model(training=False)
            logger.info("Created evaluation model")
        return self._eval_model

    @property
    def train_model(self):
        if not self._train_model:
            self._train_model = self.init_model(training=True)
            logger.info("Created training model")
        return self._train_model

    def init_sample(self, x: I) -> BeamSearchSample[I, T]:
        try:
            input_enc = self.input_tokenizer(
                x.to_str(**self.input_kwargs),
                max_length=self.max_input_length,
                padding="max_length",
                return_length=True,
                return_tensors="pt",
                truncation=False,
            )
            if input_enc.pop("length") > self.max_input_length:
                raise Exception("Exceeding padding length")
            return BeamSearchSample(
                inp=x,
                inp_enc=input_enc,
                id=None,
                name=x.name if hasattr(x, "name") else None,
            )
        except Exception as err:
            return BeamSearchSample(
                inp=x,
                inp_enc_err=str(err),
                id=None,
                name=x.name if hasattr(x, "name") else None,
            )

    def init_supervised_sample(self, x: Supervised[I, T]) -> BeamSearchLabeledSample[I, T]:
        try:
            input_enc = self.input_tokenizer(
                x.input.to_str(**self.input_kwargs),
                max_length=self.max_input_length,
                padding="max_length",
                return_length=True,
                return_tensors="pt",
                truncation=False,
            )
            if input_enc.pop("length") > self.max_input_length:
                raise Exception("Exceeding input padding length")
            sample = BeamSearchLabeledSample(
                inp=x.input,
                inp_enc=input_enc,
                id=None,
                tar=x.target,
                name=x.name if hasattr(x, "name") else None,
            )
        except Exception as err:
            return BeamSearchLabeledSample(
                inp=x.input,
                inp_enc_err=str(err),
                id=None,
                tar=x.target,
                name=x.name if hasattr(x, "name") else None,
            )

        try:
            target_enc = self.target_tokenizer(
                x.target.to_str(**self.target_kwargs),
                max_length=self.max_target_length,
                padding="max_length",
                return_length=True,
                return_tensors="pt",
                truncation=False,
            )
            if target_enc.pop("length") > self.max_target_length:
                raise Exception("Exceeding target padding length")
            # TODO inputs ids hack (because of metric)
            sample.tar_enc = target_enc["input_ids"][0]
        except Exception as err:
            sample.tar_enc_err = str(err)

        return sample

    def eval_init_sample(self, sample, **kwargs):
        start = time.time()
        preds = self.eval_model.generate(
            input_ids=sample.inp_enc["input_ids"],
            attention_mask=sample.inp_enc["attention_mask"],
            do_sample=False,
            max_length=self.max_target_length,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size,
            return_dict_in_generate=True,
        )
        for beam_id, beam in enumerate(preds["sequences"]):
            try:
                pred_str = self.target_tokenizer.decode(beam, skip_special_tokens=True)
                pred = self.target_dtype.from_str(pred_str, **self.target_kwargs)
                end = time.time()
                # TODO start id hack (because of metric)
                sample.add_beam(Beam(id=beam_id, pred=pred, pred_enc=beam[1:], time=end - start))
            except Exception as err:
                end = time.time()
                # TODO start id hack (because of metric)
                sample.add_beam(
                    Beam(id=beam_id, pred_enc=beam[1:], pred_dec_err=str(err), time=start - end)
                )
        return sample

    def eval_supervised_sample(
        self, x: Supervised[I, T], **kwargs
    ) -> BeamSearchLabeledSample[I, T]:
        sample = self.init_supervised_sample(x)

        if sample.inp_enc is None or sample.tar_enc is None:
            return sample

        return self.eval_init_sample(sample)

    def eval_sample(self, x: I, training: bool = False, **kwargs) -> BeamSearchSample[I, T]:
        sample = self.init_sample(x)

        if sample.inp_enc is None:
            return sample

        return self.eval_init_sample(sample)

    def eval(
        self,
        dataset: Dataset[I],
        batch_size: int = 32,
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        **kwargs,
    ) -> Generator[BeamSearchSample[I, T], None, None]:
        if metric is None:
            metric = NullMetric()
        if callbacks is None:
            callbacks = []

        logger.info(f"Evaluating dataset {dataset.name}")
        pbar = tqdm(desc="Evaluated samples", unit="sample")

        for x in dataset.generator():
            sample = self.eval_sample(x)
            metric.add(sample)
            [callback.add(sample) for callback in callbacks]
            pbar.update()
            pbar.set_postfix(metric.compute_dict())
            if sample.pred is not None or return_errors:
                yield sample

        pbar.close()

    def eval_supervised(
        self,
        dataset: Dataset[Supervised[I, T]],
        batch_size: int = 32,
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        **kwargs,
    ) -> Generator[BeamSearchLabeledSample[I, T], None, None]:
        if metric is None:
            metric = NullMetric()
        if callbacks is None:
            callbacks = []

        logger.info(f"Evaluating dataset {dataset.name}")
        pbar = tqdm(desc="Evaluated samples", unit="sample")

        for x in dataset.generator():
            sample = self.eval_supervised_sample(x)
            metric.add(sample)
            [callback.add(sample) for callback in callbacks]
            pbar.update()
            pbar.set_postfix(metric.compute_dict())
            if sample.pred is not None or return_errors:
                yield sample

        pbar.close()

    def init_model(self, training: bool = False, **kwargs):
        if self.hf_checkpoint_name:
            return AutoModelForSeq2SeqLM.from_pretrained(self.hf_checkpoint_name)
        elif self.checkpoint_name:
            return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_path)
        else:
            config = AutoConfig.from_pretrained(self.model_config)
            return AutoModelForSeq2SeqLM.from_config(config)

    def get_hf_dataset_supervised(
        self, dataset: Dataset[Supervised[I, T]], return_error_callbacks: bool = False
    ):
        input_token_errs: Dict[str, int] = {}
        target_token_errs: Dict[str, int] = {}

        def generator():
            for sample in dataset.generator():
                inputs = self.input_tokenizer(
                    sample.input.to_str(**self.input_kwargs),
                    max_length=self.max_input_length,
                    padding="max_length",
                    return_length=True,
                    return_tensors="pt",
                    truncation=False,
                )
                if inputs.pop("length") > self.max_input_length:
                    input_token_errs["max_input_length"] = (
                        input_token_errs.get("max_input_length", 0) + 1
                    )
                    continue
                with self.target_tokenizer.as_target_tokenizer():
                    labels = self.target_tokenizer(
                        sample.target.to_str(**self.target_kwargs),
                        max_length=self.max_target_length,
                        padding="max_length",
                        return_length=True,
                        return_tensors="pt",
                        truncation=False,
                    )
                if labels.pop("length") > self.max_target_length:
                    target_token_errs["max_target_length"] = (
                        target_token_errs.get("max_target_length", 0) + 1
                    )
                    continue
                inputs["labels"] = labels["input_ids"]
                # TODO drop prepended batch axis
                for k, v in inputs.items():
                    inputs[k] = v[0]
                yield inputs

        if return_error_callbacks:
            log_dir = os.path.join(self.local_path, "tokenizer-errors")
            log_dir = os.path.join(log_dir, dataset.name.replace("/", "-"))
            input_errs_callback = TokenizationErrorCallback(
                name=f"{dataset.name} inputs",
                errors=input_token_errs,
                log_dir=log_dir,
                filename="input-errors",
            )
            target_errs_callback = TokenizationErrorCallback(
                name=f"{dataset.name} targets",
                errors=target_token_errs,
                log_dir=log_dir,
                filename="target-errors",
            )
            return IterableDataset.from_generator(generator), [
                input_errs_callback,
                target_errs_callback,
            ]
        else:
            return IterableDataset.from_generator(generator)

    @staticmethod
    def default_metric() -> Metric:
        return MetricGroup([Counter(), EvalErrCounter()])

    @staticmethod
    def default_supervised_metric() -> Metric:
        return MetricGroup(
            [
                Acc(pad_same_length=True),
                AccPerSeq(pad_same_length=True),
                Counter(),
                EvalSupervisedErrCounter(),
            ]
        )
