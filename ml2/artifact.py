import logging
import os
import wandb

from .gcp_bucket import ML2_BUCKET, download_path, latest_version, path_exists, upload_path
from .globals import LOCAL_STORAGE_DIR, WANDB_ENTITY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Artifact(object):

    ALIASES = {}
    BUCKET_DIR = ''
    LOCAL_DIR = LOCAL_STORAGE_DIR
    WANDB_PROJECT = ''
    WANDB_TYPE = ''

    def __init__(self, name: str = None, metadata: dict = None):
        self.name = name
        self.metadata = metadata if metadata else {}

    def add_to_wandb(self, name: str) -> None:
        run = wandb.init(entity=WANDB_ENTITY,
                         name=f'add-{name}',
                         project=self.WANDB_PROJECT)
        wandb_artifact = wandb.Artifact(name,
                                        type=self.WANDB_TYPE,
                                        metadata=self.metadata)
        wandb_artifact.add_reference(
            f'gs://{ML2_BUCKET}/{self.bucket_path(name)}')
        run.log_artifact(wandb_artifact)

    def save_to_path(self, path: str) -> None:
        """Saves the artifact to a file or directory path"""
        raise NotImplementedError()

    def save(self,
             name: str,
             auto_version: bool = False,
             upload: bool = False,
             overwrite_local: bool = False,
             overwrite_bucket: bool = False,
             add_to_wandb: bool = False) -> None:

        if auto_version:
            version = latest_version(self.BUCKET_DIR, name) + 1
            name += f'-{version}'

        if os.path.exists(self.local_path(name)) and not overwrite_local:
            answer = input(
                f"Artifact {name} exists locally. Do you want to overwrite it? [y/N]"
            )
            if not (answer.lower() == 'y' or answer.lower() == 'yes'):
                return

        self.save_to_path(self.local_path(name))

        if upload:
            self.upload(name, overwrite_bucket)

        if add_to_wandb:
            self.add_to_wandb(name)

    @classmethod
    def bucket_path(cls, name: str):
        return os.path.join(cls.BUCKET_DIR, name)

    @classmethod
    def download(cls, name: str, overwrite: bool = False) -> None:
        if name in cls.ALIASES:
            name = cls.ALIASES[name]
        if os.path.exists(cls.local_path(name)):
            if overwrite:
                logger.info('Overwriting local %s %s', cls.WANDB_TYPE, name)
            else:
                logger.info('Found %s %s locally', cls.WANDB_TYPE, name)
                return
        logger.info('Downloading %s', cls.WANDB_TYPE)
        download_path(cls.bucket_path(name), cls.local_path(name))
        logger.info('Downloaded %s %s to %s', cls.WANDB_TYPE, name,
                    cls.local_path(name))

    @classmethod
    def local_path(cls, name: str):
        return os.path.join(cls.LOCAL_DIR, cls.bucket_path(name))

    @classmethod
    def load_from_path(cls, path: str):
        raise NotImplementedError()

    @classmethod
    def load(cls, name: str, overwrite: bool = False):
        if name in cls.ALIASES:
            name = cls.ALIASES[name]
        cls.download(name, overwrite)
        artifact = cls.load_from_path(cls.local_path(name))
        artifact.name = name
        return artifact

    @classmethod
    def upload(cls, name: str, overwrite: bool = False):
        if path_exists(cls.bucket_path(name)) and not overwrite:
            answer = input(
                f"Artifact {name} already exists in the bucket. Do you want to overwrite it? [y/N]"
            )
            if not (answer.lower() == 'y' or answer.lower() == 'yes'):
                return
            else:
                logger.info(f'Overwriting artifact {name} in bucket')

        upload_path(cls.local_path(name), cls.bucket_path(name))
        logger.info('Uploaded artifact %s to %s', name,
                    f'gs://{ML2_BUCKET}/{cls.bucket_path(name)}')
