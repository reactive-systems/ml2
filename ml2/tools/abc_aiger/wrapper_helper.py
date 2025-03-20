"""Helper fro runs"""

import os
import random
import subprocess
from typing import Callable, TypeVar

T = TypeVar("T")


class RunOutput:
    def __init__(self, stdouts: list[bytes], stderrs: list[bytes]):
        self.stdouts = stdouts
        self.stderrs = stderrs

    def append(self, run_output: "RunOutput"):
        self.stdouts = self.stdouts + run_output.stdouts
        self.stderrs = self.stderrs + run_output.stderrs

    def prepend(self, run_output: "RunOutput"):
        self.stdouts = run_output.stdouts + self.stdouts
        self.stderrs = run_output.stderrs + self.stderrs

    def serialize(self):
        return f"OUT: {','.join(i.decode('utf-8') for i in self.stdouts)}\nIN: {','.join(i.decode('utf-8') for i in self.stderrs)}"


class RunException(Exception, RunOutput):
    pass


def run_safe(args, timeout=None) -> tuple[subprocess.CompletedProcess, RunOutput]:
    try:
        out = subprocess.run(args, capture_output=True, timeout=timeout, check=True)
        return out, RunOutput([out.stdout], [out.stderr])
    except subprocess.TimeoutExpired as exc:
        raise RunException([b""], [b"Timeout"]) from exc
    except subprocess.CalledProcessError as e:
        raise RunException([e.stdout], [e.stderr]) from e


def run_safe_wrapper(
    fn: Callable[[], tuple[T, RunOutput]], run_out: RunOutput
) -> tuple[T, RunOutput]:
    try:
        ret, out = fn()
        run_out.append(out)
        return ret, out
    except RunException as e:
        e.prepend(run_out)
        raise e


def hash_folder(file_ext: str, temp_dir="/tmp") -> str:
    hash_random = random.getrandbits(128)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return os.path.join(temp_dir, f"aiger_input_{hash_random:032x}{file_ext}")


def change_file_ext(path: str, file_ext: str) -> str:
    return ".".join(path.split(".")[:-1]) + file_ext
