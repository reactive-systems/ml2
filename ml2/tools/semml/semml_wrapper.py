"""Wrapper for calling SemML"""

import logging
import subprocess
from typing import Any, Dict

from ml2.ltl.ltl_syn.ltl_syn_status import LTLSynStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEMML_BIN_PATH = "/semml/semml-dev/bin/semml"


def semml_wrapper_str(
    formula_str, ins_str, outs_str, parameters: Dict[str, str], timeout=None, system_format="aiger"
) -> Dict[str, Any]:
    try:
        args = [SEMML_BIN_PATH, "semmlMain", "--formula", formula_str]
        if ins_str:
            args.append(f"--env={ins_str}")
        if outs_str:
            args.append(f"--sys={outs_str}")
        for k, i in parameters.items():
            args.append(k)
            if len(str(i)) != 0:
                args.append(str(i))
        logger.debug("subprocess args: %s", args)
        print("subprocess call", " ".join(args))
        out = subprocess.run(args, capture_output=True, timeout=timeout)

        logger.debug("SemML returncode: %s", out.returncode)
        semml_stdout = out.stdout.decode("utf-8")
        logger.debug("SemML stdout: %s", semml_stdout)
        semml_stdout_lines = semml_stdout.splitlines()
        if (
            out.returncode == 0
            and semml_stdout_lines[0] == "REALIZABLE"
            and semml_stdout_lines[1] == "// start_of_aiger"
            and semml_stdout_lines[-1] == "// end_of_aiger"
        ):
            logger.debug("realizable")
            system = "\n".join(semml_stdout_lines[2:-2])
            if system_format == "aiger":
                logger.debug("AIGER circuit: %s", system)
                return {"status": LTLSynStatus("realizable"), "circuit": system}
        if out.returncode == 0 and semml_stdout_lines[0] == "UNREALIZABLE":
            logger.debug("unrealizable")
            # Doesn't return a circuit yet in unrealizable case
            return {"status": LTLSynStatus("unrealizable")}

    except subprocess.TimeoutExpired:
        logger.debug("SemMl timeout")
        return {"status": LTLSynStatus("timeout")}
    except subprocess.CalledProcessError as e:
        logger.error("subprocess called process error")
        semml_stdout = e.stdout.decode("utf-8")
        semml_stderr = e.stderr.decode("utf-8")
        return {
            "status": LTLSynStatus("error"),
            "message": "OUT:\n" + semml_stdout + "ERR:\n" + semml_stderr,
        }
    except Exception as error:
        logger.critical(error)
        return {"status": LTLSynStatus("error"), "message": str(error)}
    logger.debug("SemML error")
    return {"status": LTLSynStatus("error"), "message": "OUT:\n" + semml_stdout}
