"""Wrapper for calling Strix"""

import logging
import subprocess

from ml2.ltl.ltl_spec import LTLSpec
from ml2.ltl.ltl_syn.ltl_syn_status import LTLSynStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRIX_BIN_PATH = "/strix/bin/strix"


def strix_wrapper_dict(
    problem: dict, minimize_aiger=False, minimize_mealy=False, threads=None, timeout=None
):
    ltl_spec = LTLSpec.from_dict(problem)
    return strix_wrapper_str(
        ltl_spec.formula_str,
        ltl_spec.input_str,
        ltl_spec.output_str,
        minimize_aiger,
        minimize_mealy,
        threads,
        timeout,
    )


def strix_wrapper_str(
    formula_str,
    ins_str,
    outs_str,
    minimize_aiger=False,
    minimize_mealy=False,
    threads=None,
    timeout=None,
):
    try:
        args = [STRIX_BIN_PATH, "-f", formula_str]
        if ins_str:
            args.append(f"--ins={ins_str}")
        if outs_str:
            args.append(f"--outs={outs_str}")
        if minimize_aiger:
            args.append("--auto")
        if minimize_mealy:
            args.append("--minimize")
        if threads:
            args.append("--threads")
            args.append(str(threads))
        logger.debug("subprocess args: %s", args)
        out = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("Strix timeout")
        return {"status": LTLSynStatus.TIMEOUT, "circuit": ""}
    except subprocess.CalledProcessError:
        logger.error("subprocess called process error")
        return {"status": LTLSynStatus.ERROR, "circuit": ""}
    except Exception as error:
        logger.critical(error)
        return {"status": LTLSynStatus.ERROR, "circuit": ""}
    logger.debug("Strix returncode: %s", out.returncode)
    strix_stdout = out.stdout.decode("utf-8")
    logger.debug("Strix stdout: %s", strix_stdout)
    strix_stdout_lines = strix_stdout.splitlines()
    if out.returncode == 0 and strix_stdout_lines[0] == "REALIZABLE":
        logger.debug("realizable")
        aiger_circuit = "\n".join(strix_stdout_lines[1:])
        logger.debug("AIGER circuit: %s", aiger_circuit)
        return {"status": LTLSynStatus.REALIZABLE, "circuit": aiger_circuit}
    if out.returncode == 0 and strix_stdout_lines[0] == "UNREALIZABLE":
        logger.debug("unrealizable")
        aiger_circuit = "\n".join(strix_stdout_lines[1:])
        logger.debug("AIGER circuit: %s", aiger_circuit)
        return {"status": LTLSynStatus.UNREALIZABLE, "circuit": aiger_circuit}
    logger.debug("Strix error")
    return {"status": LTLSynStatus.ERROR, "message": strix_stdout, "circuit": ""}
