"""Experiment class"""

import functools
import logging
import os
from typing import Any, Dict, List

from ..artifact import Artifact
from ..pipelines import EvalTask
from ..registry import register_type
from ..train import Trainer
from ..utils.dict_utils import map_nested_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class Experiment(Artifact):
    WANDB_TYPE = "experiment"

    def __init__(
        self,
        name: str,
        trainer: Trainer = None,
        eval_configs: List[dict] = None,
        group: str = None,
        upload_experiment: bool = False,
        **kwargs,
    ) -> None:
        self.trainer = trainer
        self.eval_configs = eval_configs if eval_configs is not None else []
        self.group = group
        self.upload_experiment = upload_experiment

        super().__init__(name=name, **kwargs)

        slurm_environ = dict((k, v) for k, v in os.environ.items() if k.startswith("SLURM"))
        if len(slurm_environ) != 0:
            self.metadata["slurm"] = slurm_environ

    @property
    def eval_tasks(self) -> List[EvalTask]:
        eval_tasks = []
        for i, eval_config in enumerate(self.eval_configs):
            pipeline_config = eval_config["pipeline"]

            # every string starting with dollar sign is evaluated as an attribute of the experiment class
            def eval_attr(x):
                if isinstance(x, str) and x.startswith("$"):
                    # allow nested / dotted attribute paths
                    attr_path = x[1:]
                    return functools.reduce(getattr, [self] + attr_path.split("."))
                else:
                    return x

            if isinstance(pipeline_config, dict):
                map_nested_dict(eval_attr, pipeline_config)
            else:
                pipeline_config = eval_attr(pipeline_config)

            from ..loading import get_artifact_type

            pipeline_type = get_artifact_type(eval_config["pipeline"])
            for i, (delimiter, expanded_eval_config) in enumerate(
                pipeline_type.expand_eval_config(eval_config)
            ):
                expanded_eval_config["name"] += f"/{i}"
                if "type" not in expanded_eval_config:
                    raise Exception("Type not specified in eval config")

                from ..registry import type_from_str

                eval_task_type = type_from_str(expanded_eval_config["type"], bound=EvalTask)

                eval_tasks.append(eval_task_type.from_config(expanded_eval_config))

        return eval_tasks

    def get_eval_result(self, name: str):
        raise NotImplementedError()

    def run(self) -> None:
        if self.trainer is not None:
            self.trainer.train()
            self.trainer.save(
                overwrite_local=True,
                overwrite_bucket=True,
                recurse=True,
                upload=self.upload_experiment,
            )

        for eval_task in self.eval_tasks:
            try:
                eval_task.run()
                eval_task.save(upload=self.upload_experiment, overwrite_bucket=True)
            except Exception as err:
                raise err
        self.save(upload=self.upload_experiment, overwrite_bucket=True)

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_eval_names(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "evaluation" in config:
                config["eval_configs"] = config.pop("evaluation")

        def preprocess_hierarchy(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if (
                "trainer" in config
                and isinstance(config["trainer"], dict)
                and "pipeline" in config
            ):
                config["trainer"]["pipeline"] = config.pop("pipeline")

        def preprocess_upload(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "upload" in config:
                config["upload_experiment"] = config.pop("upload")

        def preprocess_eval_configs(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "eval_configs" in config and isinstance(config["eval_configs"], list):
                for i, eval_config in enumerate(config["eval_configs"]):
                    if "project" not in eval_config:
                        eval_config["project"] = config["project"]
                    if "name" not in eval_config:
                        eval_config["name"] = config["name"] + f"/eval/{i}"
                    if "pipeline" not in eval_config and config["trainer"] is None:
                        raise Exception(
                            "Neither pipeline for eval task nor trainer for experiment specified in config"
                        )
                    if "pipeline" not in eval_config:
                        eval_config["pipeline"] = config["trainer"].pipeline.full_name
                    pipeline_config = eval_config["pipeline"]
                    if (
                        isinstance(pipeline_config, dict)
                        and "type" not in pipeline_config
                        and "name" not in pipeline_config
                        and "base" not in pipeline_config
                    ):
                        pipeline_config["base"] = config["trainer"].pipeline.full_name
                        # pipeline_config["name"] = eval_config["name"] + "/pipe"

        return (
            [preprocess_eval_names, preprocess_hierarchy]
            + super().config_preprocessors()
            + [preprocess_eval_configs, preprocess_upload]
        )
