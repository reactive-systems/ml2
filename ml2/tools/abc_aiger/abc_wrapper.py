"""Wrapper for calling ABC"""

import logging
import os

from ml2.aiger import AIGERCircuit
from ml2.tools.abc_aiger.aiger_wrapper import (
    aag_file_to_aig_file,
    aig_file_to_aag,
    aig_file_to_aag_file,
)
from ml2.tools.abc_aiger.wrapper_helper import RunOutput, hash_folder, run_safe, run_safe_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ABC_BIN = "/abc/abc"

KNOR_COMPRESS_COMMANDS = [
    "dc2",
    "dc2",
    "drw -z -r -C 100 -N 10000",
    "drf -z -C 10000 -K 15",
    "balance",
    "resub -K 6",
    "rewrite",
    "resub -K 6 -N 2",
    "refactor",
    "resub -K 8",
    "balance",
    "resub -K 8 -N 2",
    "rewrite",
    "resub -K 10",
    "rewrite -z",
    "resub -K 10 -N 2",
    "balance",
    "resub -K 12",
    "refactor -z",
    "resub -K 12 -N 2",
    "balance",
    "rewrite -z",
    "balance",
]


def run_abc(circ_path: str, abc_commands: str, timeout=None) -> tuple[str, RunOutput]:
    args = [ABC_BIN, "-q", abc_commands, "-o", circ_path, circ_path]
    _, out = run_safe(args, timeout)
    return circ_path, out


def simplify_file_to_aag(
    aag_path: str, command_sequence: list[str] | None = None, timeout=None
) -> tuple[list[AIGERCircuit], RunOutput]:
    if command_sequence is None:
        command_sequence = KNOR_COMPRESS_COMMANDS
    run_out = RunOutput([], [])
    aag_hist: list[AIGERCircuit] = []
    aig_path, run_out = run_safe_wrapper(lambda: aag_file_to_aig_file(aag_path, timeout), run_out)
    while True:
        aag, run_out = run_safe_wrapper(lambda: aig_file_to_aag(aig_path, timeout), run_out)
        if len(aag_hist) > 0 and aag_hist[-1].num_gates <= aag.num_gates:
            break
        aag_hist.append(aag)
        for abc_command in command_sequence:
            _, run_out = run_safe_wrapper(lambda: run_abc(aig_path, abc_command, timeout), run_out)
    aag_path, run_out = run_safe_wrapper(lambda: aig_file_to_aag_file(aig_path, timeout), run_out)
    os.remove(aig_path)
    return aag_hist, run_out


def simplify(
    aag: AIGERCircuit,
    command_sequence: list[str] | None = None,
    timeout=None,
    temp_dir="/tmp",
) -> tuple[list[AIGERCircuit], RunOutput]:
    if command_sequence is None or len(command_sequence) == 0:
        command_sequence = KNOR_COMPRESS_COMMANDS
    aag_path = hash_folder(".aag", temp_dir)
    aag.to_file(aag_path)
    return simplify_file_to_aag(aag_path, command_sequence, timeout)
