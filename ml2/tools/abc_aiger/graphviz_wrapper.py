"""Wrapper for calling AIGER"""

import base64
import logging
import os

from ml2.aiger import AIGERCircuit
from ml2.tools.abc_aiger.wrapper_helper import change_file_ext, hash_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dot_file_to_svg_file(dot_path) -> str:
    import pydot

    svg_path = change_file_ext(dot_path, ".svg")
    graph = pydot.graph_from_dot_file(dot_path)[0]
    graph.write_svg(svg_path)
    return svg_path


def dot_file_to_svg(dot_path: str) -> str:
    svg_path = dot_file_to_svg_file(dot_path)
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    os.remove(svg_path)
    return svg


def dot_to_svg(dot: str, temp_dir="/tmp") -> str:
    dot_path = hash_folder(".dot", temp_dir)
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot)
    svg = dot_file_to_svg(dot_path)
    os.remove(dot_path)
    return svg


def dot_file_to_png_file(dot_path) -> str:
    import pydot

    png_path = change_file_ext(dot_path, ".png")
    graph = pydot.graph_from_dot_file(dot_path)[0]
    graph.write_png(png_path)
    return png_path


def dot_file_to_png(dot_path: str) -> str:
    png_path = dot_file_to_png_file(dot_path)
    with open(png_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    os.remove(png_path)
    return base64_str


def dot_to_png(dot: str, temp_dir="/tmp") -> str:
    dot_path = hash_folder(".dot", temp_dir)
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot)
    png = dot_file_to_png(dot_path)
    os.remove(dot_path)
    return png
