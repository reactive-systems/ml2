"""Artifact"""

import json
import logging
import os
from typing import Any, Dict, Union, final

import wandb
from wandb import CommError

from .configurable import Configurable
from .gcp_bucket import (
    ML2_BUCKET,
    auto_version,
    create_latest_version_dummy,
    download_path,
    fetch_file,
    latest_version,
    path_exists,
    upload_path,
)
from .globals import LOCAL_STORAGE_DIR, WANDB_ENTITY
from .utils.typing_utils import is_subclass_generic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Artifact(Configurable):
    ALIASES: Dict[str, str] = {}
    WANDB_TYPE = "artifact"

    def __init__(
        self,
        name: str,
        project: str = None,
        auto_version: bool = False,
        history: dict = None,
        metadata: dict = None,
    ):
        self.name = name

        if project is None:
            logger.warning(
                f"Project not specified on construction of {self.__class__.__name__} {name}"
            )
        self.project = project

        if auto_version:
            version = latest_version(bucket_dir=self.project if self.project else "", name=self.name) + 1  # type: ignore
            self.name += f"-{version}"  # type: ignore

        self.history = history if history is not None else {}
        self.metadata = metadata if metadata is not None else {}

    @final
    @property
    def bucket_path(self) -> str:
        return self.bucket_path_from_name(name=self.name, project=self.project)

    def config_postprocessors(self) -> list:
        def postprocess_auto_version(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("auto_version", None)
            annotations.pop("auto_version", None)

        def postprocess_history(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("history", None)
            annotations.pop("history", None)

        def postprocess_metadata(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("metadata", None)
            annotations.pop("metadata", None)

        return [
            postprocess_auto_version,
            postprocess_history,
            postprocess_metadata,
        ] + super().config_postprocessors()

    @property
    def full_name(self) -> str:
        if self.project is None:
            logger.warning(f"Accessing full name of artifact {self.name} with project set to None")
            return self.name
        else:
            return self.project + "/" + self.name

    def get_config(self, **kwargs) -> dict:
        return self.history if self.history else super().get_config(**kwargs)

    @final
    @property
    def local_path(self) -> str:
        return self.local_path_from_name(name=self.name, project=self.project)

    def save_to_path(self, path: str) -> None:
        """Saves the artifact metadata to a file or directory path"""
        if self.metadata:
            metadata_filepath = os.path.join(path, "metadata.json")
            with open(metadata_filepath, "w") as metadata_file:
                json.dump(self.metadata, metadata_file, indent=2, sort_keys=True)

    def save(
        self,
        add_to_wandb: bool = False,
        overwrite_bucket: bool = False,
        overwrite_local: bool = False,
        recurse: bool = False,
        upload: bool = False,
        **kwargs,
    ) -> None:
        config_path = os.path.join(self.local_path, "config.json")
        # check config_path instead of local_path due to child artifacts, e.g., t-1/pipe
        if os.path.exists(config_path) and not overwrite_local:
            answer = input(
                f"Artifact {self.name} exists locally. Do you want to overwrite it? [y/N]"
            )
            if not (answer.lower() == "y" or answer.lower() == "yes"):
                logger.info(f"Existing artifact {self.name} was not overwritten")
                return

        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
            logger.info("Created directory %s", self.local_path)

        config = self.get_config()
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)
        logger.info("Written config to %s", config_path)

        if "base" not in config:
            self.save_to_path(path=self.local_path, **kwargs)

        if upload:
            self.upload(name=self.name, project=self.project, overwrite=overwrite_bucket)

        if add_to_wandb:
            self.add_to_wandb(name=self.name, project=self.project, overwrite=overwrite_bucket)

    @classmethod
    def add_to_wandb(cls, name: str, project: str = None, overwrite: bool = False) -> None:
        if project is None:
            raise ValueError(
                f"Can not add artifact {name} to Weights and Biases without explicitly specifying project"
            )

        # using bucket path to name artifact in Weights and Biases
        bucket_path = cls.bucket_path_from_name(name=name, project=project)

        try:
            api = wandb.Api()
            api.artifact(f"{WANDB_ENTITY}/{bucket_path}:latest")
        except CommError:
            # Weights and Biases artifact does not exists yet
            pass
        else:
            # Weights and Biases artifact already exists
            logger.info("Artifact %s has already been added to Weight and Biases", bucket_path)
            if overwrite:
                logger.info("Adding artifact as a new version...")
            else:
                logger.info(
                    "If you would like to add the artifact as a new version set the overwrite to True"
                )
                return

        artifact = cls.load(name=name, project=project)
        run = wandb.init(entity=WANDB_ENTITY, name=f"add-{name}", project=project)
        # wandb artifact name can only contain alphanumeric characters, dashes, underscores and dots
        wandb_artifact = wandb.Artifact(
            name.replace("/", "-"), type=cls.WANDB_TYPE, metadata=artifact.metadata
        )
        wandb_artifact.add_reference(f"gs://{ML2_BUCKET}/{bucket_path}")
        run.log_artifact(wandb_artifact)  # type: ignore

    @classmethod
    def bucket_path_from_name(cls, name: str, project: str = None) -> str:
        return name if project is None else os.path.join(project, name)

    @final
    @classmethod
    def config_path_from_artifact_path(cls, artifact_path: str) -> str:
        return os.path.join(artifact_path, "config.json")

    @classmethod
    def download(cls, name: str, project: str = None, overwrite: bool = False) -> str:
        if name in cls.ALIASES:
            name = cls.ALIASES[name]
        local_path = cls.local_path_from_name(name=name, project=project)
        config_path = cls.config_path_from_artifact_path(local_path)
        # check config_path instead of local_path due to child artifacts, e.g., dataset/test
        if os.path.exists(config_path):
            if overwrite:
                logger.info("Overwriting local %s %s", cls.WANDB_TYPE, name)
            else:
                logger.debug("Found %s %s locally", cls.WANDB_TYPE, name)
                return local_path

        bucket_path = cls.bucket_path_from_name(name=name, project=project)
        if not path_exists(bucket_path):
            raise ValueError(f"{cls.__name__} {name} does not exist in bucket")
        logger.info("Downloading %s", cls.WANDB_TYPE)
        download_path(bucket_path=bucket_path, local_path=local_path)
        logger.debug("Downloaded %s %s to %s", cls.WANDB_TYPE, name, local_path)
        return local_path

    @final
    @classmethod
    def fetch_config(cls, name: str, project: str = None) -> dict:
        local_path = cls.local_path_from_name(name=name, project=project)
        local_config_path = cls.config_path_from_artifact_path(local_path)
        if os.path.exists(local_config_path):
            with open(local_config_path, "r") as local_config_file:
                return json.load(local_config_file)

        bucket_path = cls.bucket_path_from_name(name=name, project=project)
        bucket_config_path = cls.config_path_from_artifact_path(bucket_path)
        if path_exists(bucket_config_path):
            config_str = fetch_file(bucket_config_path)
            return json.loads(config_str)

        raise ValueError(f"Config {bucket_config_path} not found")

    @classmethod
    def load(cls, name: str, project: str = None, overwrite: bool = False, **kwargs) -> "Artifact":
        if name in cls.ALIASES:
            name = cls.ALIASES[name]
        local_path = cls.download(name=name, project=project, overwrite=overwrite)
        return cls.from_config_file(cls.config_path_from_artifact_path(local_path), **kwargs)

    @classmethod
    def local_path_from_name(cls, name: str, project: str = None) -> str:
        bucket_path = cls.bucket_path_from_name(name=name, project=project)
        return os.path.join(LOCAL_STORAGE_DIR, bucket_path)

    @classmethod
    def upload(cls, name: str, project: str = None, overwrite: bool = False) -> None:
        bucket_path = cls.bucket_path_from_name(name=name, project=project)
        if path_exists(bucket_path) and not overwrite:
            answer = input(
                f"Artifact {name} already exists in the bucket. Do you want to overwrite it? [y/N]"
            )
            if not (answer.lower() == "y" or answer.lower() == "yes"):
                return
            else:
                logger.info(f"Overwriting artifact {name} in bucket")

        local_path = cls.local_path_from_name(name=name, project=project)
        upload_path(local_path=local_path, bucket_path=bucket_path)
        logger.info("Uploaded artifact %s to %s", name, f"gs://{ML2_BUCKET}/{bucket_path}")

    @classmethod
    def from_config(cls, config: Union[dict, str], **kwargs) -> "Artifact":
        if isinstance(config, str):
            try:
                artifact = cls.load(name=config, **kwargs)
                logger.info(f"Loaded {artifact.__class__.__name__} {artifact.name}")
                return artifact
            except ValueError as bucket_err:
                try:
                    return super().from_config(config, **kwargs)
                except ValueError as local_err:
                    raise ValueError(str(bucket_err) + " and " + str(local_err))
        if isinstance(config, dict) and "base" in config:
            if not isinstance(config["base"], str):
                raise ValueError("Base argument in config is not of type string")
            # load artifact instead of fetching config because download is necessary for derived artifact
            base_artifact = cls.load(name=config.pop("base"))
            base_config = base_artifact.get_config()
            cls.update_config_with_parent(config, base_config)
        return super().from_config(config, **kwargs)

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_name(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "name" in config and config["name"] is not None and "name_prefix" in config:
                config["name"] = config.pop("name_prefix") + "/" + config["name"]

        def preprocess_auto_version(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if config.pop("auto_version", False):
                if config.get("upload", False):
                    config["name"] = create_latest_version_dummy(
                        name=config["name"], project=config["project"]
                    )
                else:
                    config["name"] = auto_version(name=config["name"], project=config["project"])

        def preprocess_metadata(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            path = cls.local_path_from_name(name=config["name"], project=config["project"])
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                if "metadata" in config and config["metadata"] is not None:
                    logger.warning("Metadata NOT loaded from existing file")
                else:
                    with open(metadata_path, "r") as metadata_file:
                        metadata = json.load(metadata_file)
                        logger.debug("Read metadata of %s", config["name"])
                        config["metadata"] = metadata

        def preprocess_nested_artifact(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            for name, value in config.items():
                if (
                    name in annotations
                    and is_subclass_generic(annotations[name], Artifact)
                    and isinstance(value, dict)
                ):
                    if "base" in value:
                        # test dropping this and add base to preprocess_nested_config in Configurable
                        if not isinstance(value["base"], str):
                            raise ValueError("Base argument in config is not of type string")
                        # load artifact instead of fetching config because download is necessary for derived artifact
                        from .loading import load_artifact

                        base_artifact = load_artifact(value.pop("base"))
                        base_config = base_artifact.get_config()
                        cls.update_config_with_parent(value, base_config)
                    if "name" not in value:
                        config[name]["name_prefix"] = config["name"]
                        if "default_name" in value:
                            config[name]["name"] = value.get("default_name")
                    value.pop("default_name", None)
                    if "project" not in value and config["project"] is not None:
                        config[name]["project"] = config["project"]

        return [
            preprocess_name,
            preprocess_auto_version,
            preprocess_metadata,
            preprocess_nested_artifact,
        ] + super().config_preprocessors()
