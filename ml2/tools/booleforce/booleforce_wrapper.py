"""BooleForce wrapper"""

import logging
import re
import subprocess

from ...prop import PropSatStatus
from ...prop.cnf import CNFAssignment, ResProof, ResProofCheckStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOLEFORCE_BIN_PATH = "/booleforce-1.3/booleforce"
TRACECHECK_BIN_PATH = "/booleforce-1.3/tracecheck"
TRACE_FILEPATH = "/tmp/trace.out"


def booleforce_sat_wrapper(filepath: str, timeout: float = None):
    try:
        args = [BOOLEFORCE_BIN_PATH, "-T", TRACE_FILEPATH, filepath]
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("booleforce timeout")
        return {"status": PropSatStatus("timeout")}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": PropSatStatus("error")}
    except Exception:
        logger.error("Unknown exception")
        return {"status": PropSatStatus("error")}
    stdout = out.stdout
    stdout_lines = stdout.split("\n")
    if out.returncode == 10 and stdout_lines[0].startswith("s SAT"):
        logger.error("\n".join(stdout_lines[1:]))
        assignment = CNFAssignment.from_str("\n".join(stdout_lines[1:]))
        return {
            "status": PropSatStatus("sat"),
            "assignment": assignment.assignment,
        }
    if out.returncode == 20 and stdout_lines[0].startswith("s UNSAT"):
        rp = ResProof.from_tracecheck_file(TRACE_FILEPATH)
        return {"status": PropSatStatus("unsat"), "res_proof": rp.pb}
    return {"status": PropSatStatus("error"), "message": stdout}


RESOLVED_PATTERN = re.compile(r"^resolved (\d+) root[s]? and (\d+) empty clause[s]?$\n")
FAILED_PATTERN = re.compile(r"^\*\*\* tracecheck: (.*)$\n")


def booleforce_trace_check_wrapper(filepath: str, timeout: float = None):
    try:
        args = [TRACECHECK_BIN_PATH, filepath]
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("booleforce timeout")
        return {"status": ResProofCheckStatus("timeout")}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": ResProofCheckStatus("error")}
    except Exception:
        logger.error("Unknown exception")
        return {"status": ResProofCheckStatus("error")}
    stdout = out.stdout
    resolved_pattern_match = RESOLVED_PATTERN.fullmatch(stdout)
    if out.returncode == 0 and resolved_pattern_match is not None:
        return {"status": ResProofCheckStatus("resolved")}
    failed_pattern_match = FAILED_PATTERN.fullmatch(stdout)
    if out.returncode == 0 and failed_pattern_match is not None:
        return {"status": ResProofCheckStatus("failed")}
    return {"status": ResProofCheckStatus("error"), "message": stdout}


def trace_check_binarize(
    proof_filepath: str, binarized_proof_filepath: str, timeout: float = None
):
    try:
        args = [TRACECHECK_BIN_PATH, "-B", binarized_proof_filepath, proof_filepath]
        logger.debug("subprocess")
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        logger.debug("finished")
    except subprocess.TimeoutExpired:
        logger.debug("booleforce timeout")
        return {"status": ResProofCheckStatus("timeout")}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": ResProofCheckStatus("error")}
    except Exception:
        logger.error("Unknown exception")
        return {"status": ResProofCheckStatus("error")}
    stdout = out.stdout
    resolved_pattern_match = RESOLVED_PATTERN.fullmatch(stdout)
    if out.returncode == 0 and resolved_pattern_match is not None:
        return {"status": ResProofCheckStatus("resolved")}
    failed_pattern_match = FAILED_PATTERN.fullmatch(stdout)
    if out.returncode == 0 and failed_pattern_match is not None:
        return {"status": ResProofCheckStatus("failed")}
    return {"status": ResProofCheckStatus("error"), "message": stdout}
