"""Utilities for checking if a library is available"""

import importlib


def is_pt_available() -> bool:
    """Check if PyTorch is available"""
    return importlib.util.find_spec("torch") is not None


def is_tf_available() -> bool:
    """Check if Tensorflow is available"""
    return importlib.util.find_spec("tensorflow") is not None


def is_ray_available() -> bool:
    """Check if Ray is available"""
    return importlib.util.find_spec("ray") is not None
