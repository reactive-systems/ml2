"""Wrapper for calling BoSy"""
import logging
import os
import subprocess

from ...ltl.ltl_syn.ltl_syn_status import LTLSynStatus
from .bosy_input import format_bosy_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bosy_wrapper_dict(problem: dict, bosy_path, timeout, optimize=False, temp_dir="/tmp"):
    problem_str = format_bosy_input(
        problem["guarantees"],
        problem["inputs"],
        problem["outputs"],
        problem.get("assumptions", None),
    )
    return bosy_wrapper_str(problem_str, bosy_path, timeout, optimize, temp_dir)


def bosy_wrapper_file(problem_file, bosy_path, timeout, optimize=False):
    try:
        args = [bosy_path, "--synthesize", str(problem_file)]
        if optimize:
            args.append("--optimize")
        logger.debug("subprocess args: %s", args)
        out = subprocess.run(args, capture_output=True, timeout=timeout, universal_newlines=True)
    except subprocess.TimeoutExpired:
        logger.debug("BoSy timeout")
        return {"status": LTLSynStatus.TIMEOUT, "circuit": ""}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": LTLSynStatus.ERROR, "circuit": ""}
    except Exception as error:
        logger.error(error)
        return {"status": LTLSynStatus.ERROR, "circuit": ""}
    logger.debug("BoSy stdout: %s", out.stdout)
    logger.debug("BoSy stderr: %s", out.stderr)
    if out.stdout == "":
        return {"status": LTLSynStatus.ERROR, "message": out.stderr, "circuit": ""}
    out_lines = out.stdout.splitlines()
    if out_lines[0] == "UNREALIZABLE":
        aiger_circuit = "\n".join(out_lines[1:-4])
        return {"status": LTLSynStatus.UNREALIZABLE, "circuit": aiger_circuit}
    elif out_lines[0] == "REALIZABLE":
        aiger_circuit = "\n".join(out_lines[1:-4])
        return {"status": LTLSynStatus.REALIZABLE, "circuit": aiger_circuit}
    else:
        return {"status": LTLSynStatus.ERROR, "message": out.stdout, "circuit": ""}


def bosy_wrapper_str(problem_str, bosy_path, timeout, optimize=False, temp_dir="/tmp"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    problem_filepath = os.path.join(temp_dir, "bosy_input.json")
    with open(problem_filepath, "w") as problem_file:
        problem_file.write(problem_str)
    result = bosy_wrapper_file(problem_filepath, bosy_path, timeout, optimize)
    # os.remove(problem_filepath)
    return result
