"""Trainer class"""

import logging
import os

import wandb

from ..artifact import Artifact
from ..globals import WANDB_ENTITY
from ..pipelines import Pipeline
from ..registry import register_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class Trainer(Artifact):
    WANDB_TYPE = "train"

    def __init__(
        self,
        pipeline: Pipeline,
        name: str = "train",
        checkpoint_name: str = None,
        log_freq: int = 1,
        stream_to_wandb: bool = False,
        wandb_name: str = None,
        wandb_run_id=None,
        **kwargs,
    ) -> None:
        self.pipeline = pipeline
        self.log_freq = log_freq
        self.stream_to_wandb = stream_to_wandb
        self.wandb_name = wandb_name if wandb_name is not None else name
        self.wandb_run_id = wandb_run_id

        super().__init__(name=name, **kwargs)

        self.checkpoint_name = (
            checkpoint_name if checkpoint_name is not None else self.full_name + "/ckpts"
        )

    @property
    def checkpoint_path(self) -> str:
        return self.local_path_from_name(name=self.checkpoint_name)

    def train(self):
        if self.stream_to_wandb:
            config = self.get_config()
            if "SLURM_JOB_ID" in os.environ:
                config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
            wandb.init(
                config=config,
                entity=WANDB_ENTITY,
                group=self.pipeline.group,
                name=self.wandb_name,
                project=self.pipeline.project,
                id=self.wandb_run_id,
                resume="auto",
            )
            self.wandb_run_id = wandb.run.id

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
            self.pipeline.save(
                add_to_wandb=add_to_wandb,
                overwrite_bucket=overwrite_bucket,
                overwrite_local=overwrite_local,
                recurse=True,
                upload=upload,
            )
