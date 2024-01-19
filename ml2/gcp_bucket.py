"""Google Cloud Platform storage bucket utility"""

import argparse
import json
import logging
import os

import pandas as pd
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from google.resumable_media import InvalidResponse

from .globals import ML2_BUCKET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_client() -> storage.Client:
    if ML2_BUCKET == "ml2-public":
        logger.info("public bucket, using anonymous client")
        return storage.Client.create_anonymous_client()
    try:
        return storage.Client()
    except DefaultCredentialsError:
        logger.info("Google Cloud credentials not found, trying anonymous client")
        return storage.Client.create_anonymous_client()
    except EnvironmentError as e:
        logger.info(e, " , trying anonymous client")
        return storage.Client.create_anonymous_client()


def create_latest_version_dummy(
    name: str, project: str = None, bucket_name: str = ML2_BUCKET
) -> str:
    success = False
    storage_client = create_client()
    bucket = storage_client.bucket(bucket_name)
    while not success:
        name = auto_version(name=name, project=project, bucket_name=bucket_name)
        # add slash to emulate folder
        bucket_path = (f"{project}/{name}" if project is not None else name) + "/"
        blob = bucket.blob(bucket_path)
        try:
            blob.upload_from_string("", if_generation_match=0)
        except InvalidResponse:
            success = False
        else:
            success = True
    return name


def latest_version(bucket_dir: str, name: str, bucket_name: str = ML2_BUCKET) -> int:
    storage_client = create_client()
    latest_version = -1
    prefix = f"{bucket_dir}/{name}-"
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        _, suffix = blob.name.split(prefix, 1)
        version_str, _ = suffix.split("/", 1)
        if version_str.isdigit() and int(version_str) > latest_version:
            latest_version = int(version_str)
    return latest_version


def auto_version(name: str, project: str = None, bucket_name: str = ML2_BUCKET) -> str:
    project = project if project else ""
    new_version = latest_version(bucket_dir=project, name=name, bucket_name=bucket_name) + 1
    if name.endswith("/"):
        return f"{name}{new_version}"
    else:
        return f"{name}-{new_version}"


def path_exists(path: str, bucket_name: str = ML2_BUCKET) -> bool:
    storage_client = create_client()
    blobs = storage_client.list_blobs(bucket_name, prefix=path)
    return any(True for _ in blobs)


def upload_path(
    local_path: str,
    bucket_path: str,
    bucket_name: str = ML2_BUCKET,
    skip_hidden_dirs: bool = True,
    skip_hidden_files: bool = True,
) -> None:
    """Uploads a file or directory to a GCP storage bucket

    Args:
        local_path: path to a local file or directory
        bucket_path: path where the file or directory is stored in the bucket
        bucket_name: name of the GCP storage bucket

    Raises:
        Exception: if the local path is not a valid path to a file or directory
    """
    storage_client = create_client()
    bucket = storage_client.bucket(bucket_name)
    if os.path.isfile(local_path):
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(local_path)
    elif os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            bucket_root = root.replace(local_path, bucket_path, 1)
            for file in files:
                if file.startswith(".") and skip_hidden_files:
                    # skip hidden files
                    continue
                blob = bucket.blob(f"{bucket_root}/{file}")
                blob.upload_from_filename(f"{root}/{file}")
            for d in dirs:
                if d.startswith(".") and skip_hidden_dirs:
                    # skip hidden directories
                    dirs.remove(d)
    else:
        raise Exception("Path %s is not a valid path to a file or directory", local_path)


def download_file(filename: str, local_filename: str, bucket_name: str = ML2_BUCKET):
    storage_client = create_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.download_to_filename(local_filename)


def fetch_file(filename: str, bucket_name: str = ML2_BUCKET) -> str:
    storage_client = create_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    return blob.download_as_string()


def download_path(
    bucket_path: str, local_path: str, recurse: bool = True, bucket_name: str = ML2_BUCKET
):
    """Downloads a file or directory from a GCP storage bucket

    Args:
        bucket_path: path that identifies a file in the bucket or a prefix that emulates a directory in the bucket
        local_path: path where the file or directory is stored locally
        recurse: whether to recurse on a directory
        bucket_name: name of the GCP storage bucket
    """
    storage_client = create_client()
    delimiter = None if recurse else "/"
    blobs = storage_client.list_blobs(bucket_name, prefix=bucket_path + "/", delimiter=delimiter)
    for blob in blobs:
        filepath = blob.name.replace(bucket_path, local_path, 1)
        file_dir, filename = os.path.split(filepath)
        if not filename:
            # if filename is empty blob is a folder
            continue
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        blob.download_to_filename(filepath)


def download_metadata(prefix: str, project: str, bucket_name: str = ML2_BUCKET) -> pd.DataFrame:
    """Constructs a dataframe containing all configs / metadata / arguments from a project with a given artifact prefix.

    Args:
        prefix (str): All artifacts, which metadata is collected must begin with the given prefix.
        project (str): the project directory where the artifacts lie in (i.e. ltl-syn)
        bucket_name (str, optional): The google cloud bucket name. Defaults to the content of global variable ML2_BUCKET.
    """

    def flatten(obj: dict, parent_string: str = "") -> dict:
        """Transforms any given dict with lists or dicts as elements to a dict with just a single depth, hence all values in this dicts are basic datatypes.

        Example:
            {
                "a": [1,2,3,4],
                "b": {
                    "c": 1,
                    "d": 2,
                }
            }
            becomes
            {
                "a.0": 1,
                "a.1": 2,
                "a.2": 3,
                "a.3": 4,
                "b.c": 1,
                "b.d": 2
            }

        Args:
            obj (dict): input dict
            parent_string (str, optional): For recursive remembering of the parent keys. Defaults to ""

        Returns:
            dict: the flattened dict
        """
        new_obj = {}
        for k in obj:
            if isinstance(obj[k], list):
                for i, v in enumerate(obj[k]):
                    new_obj[parent_string + k + "." + str(i)] = v
            elif isinstance(obj[k], dict):
                new_obj = {**new_obj, **flatten(obj[k], parent_string + k + ".")}
            else:
                new_obj[parent_string + k] = obj[k]
        return new_obj

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=project + "/" + prefix)
    series = []
    for blob in blobs:
        if blob.name.endswith("/config.json"):
            artifact = blob.name[len(project + "/") : blob.name.find("/config.json")]
            series.append(pd.Series(flatten(json.loads(blob.download_as_text())), name=artifact))
        if blob.name.endswith("/metadata.json"):
            artifact = blob.name[len(project + "/") : blob.name.find("/metadata.json")]
            series.append(pd.Series(flatten(json.loads(blob.download_as_text())), name=artifact))
        if blob.name.endswith("/args.json"):
            artifact = blob.name[len(project + "/") : blob.name.find("/args.json")]
            series.append(pd.Series(flatten(json.loads(blob.download_as_text())), name=artifact))

    dl = pd.concat(series, axis=1).T
    dl.reindex(sorted(dl.columns), axis=1)
    dl.sort_index()
    return dl


def cli() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="All artifacts, which configs / metadata / arguments are collected must begin with this prefix.",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        required=True,
        help="The project directory where the artifacts lie in (i.e. ltl-syn)",
    )
    parser.add_argument("-o", "--output", type=str, default="metadata.csv", help="output path")

    args = parser.parse_args()

    logger.info("Collecting metadata from " + args.project + "/" + args.name + "...")
    dataframe = download_metadata(args.name, args.project)

    dataframe.to_csv(args.output)
    logger.info("Saved to " + args.output)


if __name__ == "__main__":
    cli()
