"""TensorFlow Transformer pipeline test"""
import pathlib

import pytest


@pytest.fixture()
def test_config_path(request):
    path = pathlib.Path(request.node.fspath)
    return path.with_name("tf_transformer_pipeline_test_config.json")


def test_tf_transformer_pipeline(test_config_path):
    pass
