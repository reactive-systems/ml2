"""Google Cloud Platform storage bucket utility"""

import os
from google.cloud import storage

from .globals import ML2_BUCKET


def latest_version(bucket_dir: str, name: str, bucket_name: str = ML2_BUCKET):
    storage_client = storage.Client()
    latest_version = -1
    prefix = f"{bucket_dir}/{name}-"
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        _, suffix = blob.name.split(prefix, 1)
        version_str, _ = suffix.split("/", 1)
        if version_str.isdigit() and int(version_str) > latest_version:
            latest_version = int(version_str)
    return latest_version


def path_exists(path: str, bucket_name: str = ML2_BUCKET):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=path)
    return any(True for _ in blobs)


def upload_path(local_path: str, bucket_path: str, bucket_name: str = ML2_BUCKET):
    """Uploads a file or directory to a GCP storage bucket

    Args:
        local_path: path to a local file or directory
        bucket_path: path where the file or directory is stored in the bucket
        bucket_name: name of the GCP storage bucket

    Raises:
        Exception: if the local path is not a valid path to a file or directory
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if os.path.isfile(local_path):
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(local_path)
    elif os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            bucket_root = root.replace(local_path, bucket_path, 1)
            for file in files:
                if file.startswith("."):
                    # skip hidden files
                    continue
                blob = bucket.blob(f"{bucket_root}/{file}")
                blob.upload_from_filename(f"{root}/{file}")
            for d in dirs:
                if d.startswith("."):
                    # skip hidden directories
                    dirs.remove(d)
    else:
        raise Exception("Path %s is not a valid path to a file or directory", local_path)


def download_file(filename: str, local_filename: str, bucket_name: str = ML2_BUCKET):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.download_to_filename(local_filename)


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
    storage_client = storage.Client()
    delimiter = None if recurse else "/"
    blobs = storage_client.list_blobs(bucket_name, prefix=bucket_path, delimiter=delimiter)
    for blob in blobs:
        filepath = blob.name.replace(bucket_path, local_path, 1)
        file_dir, filename = os.path.split(filepath)
        if not filename:
            # if filename is empty blob is a folder
            continue
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        blob.download_to_filename(filepath)
