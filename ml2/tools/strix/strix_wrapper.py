"""Wrapper for calling Strix"""

import logging
import subprocess
from typing import Any, Dict

from ml2.ltl.ltl_syn.ltl_syn_status import LTLSynStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRIX_BIN_PATH = "/strix/target/release/strix"


def strix_wrapper_str(
    formula_str, ins_str, outs_str, parameters: Dict[str, str], timeout=None, system_format="aiger"
) -> Dict[str, Any]:
    try:
        args = [STRIX_BIN_PATH, "-f", formula_str]
        if ins_str:
            args.append(f"--ins={ins_str}")
        if outs_str:
            args.append(f"--outs={outs_str}")
        for k, i in parameters.items():
            args.append(k)
            if len(str(i)) != 0:
                args.append(str(i))
        if system_format == "aiger":
            args.append("--aiger")
        logger.debug("subprocess args: %s", args)
        # print("subprocess args: %s", args)
        out = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("Strix timeout")
        return {"status": LTLSynStatus("timeout")}
    except subprocess.CalledProcessError as e:
        logger.error("subprocess called process error")
        strix_stdout = e.stdout.decode("utf-8")
        strix_stderr = e.stderr.decode("utf-8")
        return {
            "status": LTLSynStatus("error"),
            "message": "OUT:\n" + strix_stdout + "ERR:\n" + strix_stderr,
        }
    except Exception as error:
        logger.critical(error)
        return {"status": LTLSynStatus("error"), "message": str(error)}
    logger.debug("Strix returncode: %s", out.returncode)
    strix_stdout = out.stdout.decode("utf-8")
    logger.debug("Strix stdout: %s", strix_stdout)
    strix_stdout_lines = strix_stdout.splitlines()
    if (
        "Warning: The current network has no primary outputs. Some commands may not work correctly."
        in strix_stdout_lines
    ):
        return {"status": LTLSynStatus("error"), "message": strix_stdout}
    if out.returncode == 0 and strix_stdout_lines[0] == "REALIZABLE":
        logger.debug("realizable")
        system = "\n".join(strix_stdout_lines[1:])
        if system_format == "aiger":
            logger.debug("AIGER circuit: %s", system)
            return {"status": LTLSynStatus("realizable"), "circuit": system}
        elif system_format == "mealy":
            logger.debug("Mealy Machine: %s", system)
            return {"status": LTLSynStatus("realizable"), "mealy_machine": system}
    if out.returncode == 0 and strix_stdout_lines[0] == "UNREALIZABLE":
        logger.debug("unrealizable")
        system = "\n".join(strix_stdout_lines[1:])
        if system_format == "aiger":
            logger.debug("AIGER circuit: %s", system)
            return {"status": LTLSynStatus("unrealizable"), "circuit": system}
        elif system_format == "mealy":
            logger.debug("Mealy Machine: %s", system)
            return {"status": LTLSynStatus("unrealizable"), "mealy_machine": system}
    logger.debug("Strix error")
    return {"status": LTLSynStatus("error"), "message": "OUT:\n" + strix_stdout}
