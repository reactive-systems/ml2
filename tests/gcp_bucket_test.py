"""GCP storage bucket tests"""

import pytest
from google.cloud import storage

from ml2.gcp_bucket import auto_version, create_latest_version_dummy, path_exists, upload_path
from ml2.globals import ML2_BUCKET


@pytest.mark.gcp
def test_gcp_bucket(tmp_path):
    local_project_path = tmp_path / "test-project"
    local_project_path.mkdir()

    local_folder_path = local_project_path / "test-folder"
    local_folder_path.mkdir()

    local_file_path = local_folder_path / "test-file.txt"
    local_file_path.write_text("Hello world")

    upload_path(local_path=str(local_folder_path), bucket_path="test-project/test-folder")

    assert path_exists(path="test-project/test-folder")

    assert auto_version(name="test-folder", project="test-project") == "test-folder-0"

    local_folder_path_10 = local_project_path / "test-folder-10"
    local_folder_path_10.mkdir()

    local_file_path_10 = local_folder_path_10 / "test-file.txt"
    local_file_path_10.write_text("Hello world")

    upload_path(local_path=str(local_folder_path_10), bucket_path="test-project/test-folder-10")

    assert (
        create_latest_version_dummy(name="test-folder", project="test-project") == "test-folder-11"
    )
    assert (
        create_latest_version_dummy(name="test-folder", project="test-project") == "test-folder-12"
    )

    storage_client = storage.Client()
    bucket = storage_client.bucket(ML2_BUCKET)
    for blob in bucket.list_blobs(prefix="test-project"):
        blob.delete()
