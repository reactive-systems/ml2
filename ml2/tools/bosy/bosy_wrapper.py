"""Wrapper for calling BoSy"""

import logging
import os
import random
import re
import subprocess
from typing import Any, Dict

from ...ltl.ltl_spec.ltl_spec import LTLSpec
from ...ltl.ltl_syn.ltl_syn_status import LTLSynStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bosy_wrapper_file(problem_file, bosy_path, timeout, parameters: Dict[str, str]):
    def filter_aag(el):
        match = re.search("aag|[0-9]|[a-z][0-9]", el)
        if match is not None and match.span()[0] == 0:
            return True
        else:
            return False

    try:
        args = [bosy_path, "--synthesize", str(problem_file)]
        for k, i in parameters.items():
            args.append(k)
            if len(str(i)) != 0:
                args.append(str(i))
        logger.debug("subprocess args: %s", args)
        out = subprocess.run(args, capture_output=True, timeout=timeout, universal_newlines=True)
    except subprocess.TimeoutExpired:
        logger.debug("BoSy timeout")
        return {"status": LTLSynStatus("timeout")}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": LTLSynStatus("error")}
    except Exception as error:
        logger.error(error)
        return {"status": LTLSynStatus("error")}
    logger.debug("BoSy stdout: %s", out.stdout)
    logger.debug("BoSy stderr: %s", out.stderr)
    if out.stdout == "":
        return {"status": LTLSynStatus("error"), "message": out.stderr}
    out_lines = out.stdout.splitlines()

    if out_lines[0] == "UNREALIZABLE":
        aiger_circuit = "\n".join(filter(filter_aag, out_lines))
        return {"status": LTLSynStatus("unrealizable"), "circuit": aiger_circuit}
    elif out_lines[0] == "REALIZABLE":
        aiger_circuit = "\n".join(filter(filter_aag, out_lines))
        return {"status": LTLSynStatus("realizable"), "circuit": aiger_circuit}
    else:
        return {"status": LTLSynStatus("error"), "message": out.stdout}


def bosy_wrapper_str(problem_str, bosy_path, timeout, optimize=False, temp_dir="/tmp"):
    hash = random.getrandbits(128)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    problem_filepath = os.path.join(temp_dir, "bosy_input_" + str("%032x" % hash) + ".json")
    with open(problem_filepath, "w") as problem_file:
        problem_file.write(problem_str)
    result = bosy_wrapper_file(problem_filepath, bosy_path, timeout, optimize)
    os.remove(problem_filepath)
    return result


def bosy_wrapper(
    problem: LTLSpec,
    bosy_path,
    timeout,
    parameters: Dict[str, str],
    temp_dir="/tmp",
) -> Dict[str, Any]:
    if problem.inputs is None or len(problem.inputs) == 0:
        # BoSy requires at least one input
        problem.inputs = ["i_default"]
    if problem.semantics is None:
        problem.semantics = "mealy"
    hash = random.getrandbits(128)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    problem_filepath = os.path.join(temp_dir, "bosy_input_" + str("%032x" % hash) + ".json")
    problem.to_bosy_file(problem_filepath)
    result = bosy_wrapper_file(problem_filepath, bosy_path, timeout, parameters)
    os.remove(problem_filepath)
    return result
