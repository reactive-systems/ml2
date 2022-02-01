"""Aalta wrapper"""

import logging
import subprocess

from ...ltl.ltl_sat.ltl_sat_status import LTLSatStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AALTA_BIN_PATH = "/aalta/aalta"


def aalta_wrapper_str(formula: str, evidence: bool = True, timeout: float = None):
    try:
        args = [AALTA_BIN_PATH, "-c", "-l"]
        if evidence:
            args.append("-e")
        out = subprocess.run(args, capture_output=True, input=formula, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("aalta timeout")
        return {"status": LTLSatStatus.TIMEOUT}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": LTLSatStatus.ERROR}
    except Exception:
        logger.error("Unknown exception")
        return {"status": LTLSatStatus.ERROR}
    aalta_stdout = out.stdout
    aalta_stdout_lines = aalta_stdout.split("\n")
    if out.returncode == 0 and aalta_stdout_lines[1] == "sat":
        return {"status": LTLSatStatus.SATISFIABLE, "trace": "\n".join(aalta_stdout_lines[2:])}
    if out.returncode == 0 and aalta_stdout_lines[1] == "unsat":
        return {"status": LTLSatStatus.UNSATISFIABLE}
    return {"status": LTLSatStatus.ERROR, "message": aalta_stdout}
