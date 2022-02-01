"""Limboole wrapper"""

import logging
import subprocess

from ...prop.prop_sat_status import PropSatStatus, PropValidStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMBOOLE_BIN_PATH = "/limboole1.2/limboole"


def limboole_sat_wrapper(formula: str, timeout: float = None):
    try:
        args = [LIMBOOLE_BIN_PATH, "-s"]
        out = subprocess.run(args, capture_output=True, input=formula, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("limboole timeout")
        return {"status": PropSatStatus.TIMEOUT}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": PropSatStatus.ERROR}
    except Exception:
        logger.error("Unknown exception")
        return {"status": PropSatStatus.ERROR}
    limboole_stdout = out.stdout
    limboole_stdout_lines = limboole_stdout.split("\n")
    if out.returncode == 0 and limboole_stdout_lines[0].startswith("% SAT"):
        return {
            "status": PropSatStatus.SAT,
            "assignment": {l[:-4]: int(l[-1]) for l in limboole_stdout_lines[1:-1]},
        }
    if out.returncode == 0 and limboole_stdout_lines[0].startswith("% UNSAT"):
        return {"status": PropSatStatus.UNSAT}
    return {"status": PropSatStatus.ERROR, "message": limboole_stdout}


def limboole_valid_wrapper(formula: str, timeout: float = None):
    try:
        args = [LIMBOOLE_BIN_PATH]
        out = subprocess.run(args, capture_output=True, input=formula, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("limboole timeout")
        return {"status": PropValidStatus.TIMEOUT}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": PropValidStatus.ERROR}
    except Exception:
        logger.error("Unknown exception")
        return {"status": PropValidStatus.ERROR}
    limboole_stdout = out.stdout
    limboole_stdout_lines = limboole_stdout.split("\n")
    if out.returncode == 0 and limboole_stdout_lines[0].startswith("% VALID"):
        return {"status": PropValidStatus.VALID}
    if out.returncode == 0 and limboole_stdout_lines[0].startswith("% INVALID"):
        return {
            "status": PropValidStatus.INVALID,
            "assignment": {l[:-4]: int(l[-1]) for l in limboole_stdout_lines[1:-1]},
        }
    return {"status": PropValidStatus.ERROR, "message": limboole_stdout}
