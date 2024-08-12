"""HuggingFace Seq2Seq Trainer"""

import logging
import os

import numpy as np
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ..datasets import Dataset
from ..pipelines import HFPTText2TextPipeline
from ..registry import register_type
from .trainer import Trainer


@register_type
class HFSeq2SeqTrainer(Trainer):
    def __init__(
        self,
        pipeline: HFPTText2TextPipeline,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-3,
        load_best_model_at_end: bool = False,
        log_freq: int = 1,
        lr_scheduler_type: str = "linear",
        metric_for_best_model: str = "acc_per_seq",
        save_freq: int = 10,
        save_limit: int = 1,
        steps: int = 100,
        val_freq: int = 10,
        warmup_steps: int = 0,
        weight_decay: float = 0.1,
        **kwargs,
    ):
        super().__init__(pipeline=pipeline, **kwargs)

        assert save_freq <= steps
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.load_best_model_at_end = load_best_model_at_end
        self.log_freq = log_freq
        self.lr_scheduler_type = lr_scheduler_type
        self.metric_for_best_model = metric_for_best_model
        self.save_freq = save_freq
        self.save_limit = save_limit
        self.steps = steps
        self.val_freq = val_freq
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        if self.checkpoint_name is None:
            self.checkpoint_name = self.full_name + "/ckpts"

    def train(self):
        super().train()

        train_args = Seq2SeqTrainingArguments(
            evaluation_strategy="steps",
            eval_steps=self.val_freq,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            load_best_model_at_end=self.load_best_model_at_end,
            logging_steps=self.log_freq,
            logging_strategy="steps",
            lr_scheduler_type=self.lr_scheduler_type,
            max_steps=self.steps,
            metric_for_best_model=self.metric_for_best_model,
            output_dir=self.checkpoint_path,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            report_to=["tensorboard", "wandb"] if self.stream_to_wandb else ["tensorboard"],
            run_name=self.name,
            save_steps=self.save_freq,
            save_strategy="steps",
            save_total_limit=self.save_limit,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
        )

        def metrics(eval_preds):
            logits = eval_preds.predictions[0]
            labels = eval_preds.label_ids
            preds = np.argmax(logits, axis=-1)
            acc_el = (preds == labels).astype(int)
            # ignore pad tokens
            acc = np.mean(acc_el, where=labels != 0)
            acc_per_seq = np.mean(np.fix(np.mean(acc_el, axis=-1, where=labels != 0)))
            return {"acc": acc, "acc_per_seq": acc_per_seq}

        hf_train_dataset, train_err_callbacks = self.pipeline.get_hf_dataset_supervised(
            self.train_dataset, return_error_callbacks=True
        )
        hf_val_dataset, val_err_callbacks = self.pipeline.get_hf_dataset_supervised(
            self.val_dataset, return_error_callbacks=True
        )

        trainer = Seq2SeqTrainer(
            args=train_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
            model=self.pipeline.train_model,
            compute_metrics=metrics,
        )

        # filter warning about adding the same callback class multiple times
        def callback_warning_filter(record):
            return not record.msg.startswith("You are adding a ")

        transformers_logger = logging.getLogger("transformers.trainer_callback")
        transformers_logger.addFilter(callback_warning_filter)
        for callback in train_err_callbacks + val_err_callbacks:
            trainer.add_callback(callback)
        transformers_logger.removeFilter(callback_warning_filter)

        trainer.train()

        self.pipeline.hf_checkpoint_name = None
        self.pipeline.checkpoint_name = os.path.join(
            self.checkpoint_name,
            os.path.relpath(get_last_checkpoint(self.checkpoint_path), self.checkpoint_path),
        )
        self.pipeline._eval_model = None

        if self.stream_to_wandb:
            wandb.finish()
