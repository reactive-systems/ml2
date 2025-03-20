"""Wrapper for calling AIGER"""

import logging
import os

from ml2.aiger import AIGERCircuit
from ml2.tools.abc_aiger.wrapper_helper import RunOutput, change_file_ext, hash_folder, run_safe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AIGTOAIG_BIN = "/aiger/aigtoaig"
AIGTODOT_BIN = "/aiger/aigtodot"


def aag_file_to_aig_file(aag_path, timeout=None) -> tuple[str, RunOutput]:
    aig_path = change_file_ext(aag_path, ".aig")
    args = [AIGTOAIG_BIN, aag_path, aig_path]
    _, out = run_safe(args, timeout)
    return aig_path, out


def aag_to_aig(aag: AIGERCircuit, timeout=None, temp_dir="/tmp") -> tuple[str, RunOutput]:
    aag_path = hash_folder(".aag", temp_dir)
    aag.to_file(aag_path)
    aig_path, out = aag_file_to_aig_file(aag_path, timeout)
    with open(aig_path, "r", encoding="utf-8") as f:
        aig = f.read()
    os.remove(aag_path)
    os.remove(aig_path)
    return aig, out


def aig_file_to_aag_file(aig_path, timeout=None) -> tuple[str, RunOutput]:
    aag_path = change_file_ext(aig_path, ".aag")
    args = [AIGTOAIG_BIN, aig_path, aag_path]
    _, out = run_safe(args, timeout)
    return aag_path, out


def aig_file_to_aag(aig_path: str, timeout=None) -> tuple[AIGERCircuit, RunOutput]:
    aag_path, out = aig_file_to_aag_file(aig_path, timeout=timeout)
    aag = AIGERCircuit.from_file(aag_path)
    os.remove(aag_path)
    return aag, out


def aig_to_aag(aig: str, timeout=None, temp_dir="/tmp") -> tuple[AIGERCircuit, RunOutput]:
    aig_path = hash_folder(".aig", temp_dir)
    with open(aig_path, "w", encoding="utf-8") as f:
        f.write(aig)
    aag, out = aig_file_to_aag(aig_path, timeout)
    os.remove(aig_path)
    return aag, out


def aag_file_to_dot_file(aag_path, timeout=None) -> tuple[str, RunOutput]:
    dot_path = change_file_ext(aag_path, ".dot")
    args = [AIGTODOT_BIN, aag_path, dot_path]
    _, out = run_safe(args, timeout)
    return dot_path, out


def aag_to_dot(aag: AIGERCircuit, timeout=None, temp_dir="/tmp") -> tuple[str, RunOutput]:
    aag_path = hash_folder(".aag", temp_dir)
    aag.to_file(aag_path)
    dot_path, out = aag_file_to_dot_file(aag_path, timeout)
    with open(dot_path, "r", encoding="utf-8") as f:
        dot = f.read()
    os.remove(aag_path)
    os.remove(dot_path)
    return dot, out
