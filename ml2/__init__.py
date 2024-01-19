"""
ML2 - Machine Learning for Mathematics and Logics
"""

__version__ = "0.2.0"

from . import (
    aiger,
    datasets,
    dtypes,
    grpc,
    layers,
    ltl,
    models,
    optim,
    prop,
    tools,
    trace,
)
from .loading import load_artifact as load
from .utils import is_pt_available, is_ray_available, is_tf_available

if is_ray_available():
    from . import data_gen

if is_pt_available() and is_tf_available():
    from . import experiment, pipelines, tokenizers, train
