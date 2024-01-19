"""Wrapper for model checking script from Strix that uses the nuXmv model checker"""

import logging
import os
import subprocess
from typing import Optional, Tuple

from ...aiger.aiger_circuit import AIGERCircuit
from ...ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ...ltl.ltl_spec.ltl_spec import LTLSpec


def nuxmv_wrapper(
    spec: LTLSpec,
    circuit: AIGERCircuit,
    realizable: bool,
    strix_path: str,
    temp_dir: str,
    timeout: Optional[float] = None,
) -> Tuple[LTLMCStatus, str]:
    timeout = timeout if timeout is not None else 10.0
    temp_dir = os.path.join(temp_dir, "nuxmv")
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    spec.to_file(temp_dir, "spec.tlsf", file_format="tlsf")
    circuit_filepath = os.path.join(temp_dir, "circuit.aag")
    # nuXmv requires newline character at the end of the aiger file
    circuit_str = circuit.to_str()
    if not circuit_str.endswith("\n"):
        circuit_str = circuit_str + "\n"
    with open(circuit_filepath, "w") as aiger_file:
        aiger_file.write(circuit_str)
    verify_script_path = os.path.join(strix_path, "scripts/verify.sh")
    try:
        args = [verify_script_path, circuit_filepath, os.path.join(temp_dir, "spec.tlsf")]
        if realizable:
            args.append("REALIZABLE")
        else:
            args.append("UNREALIZABLE")
        args.append(str(timeout))
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.debug("subprocess timeout")
        status = LTLMCStatus("timeout")
        detailed_status = (
            status.token().upper() + ":\n" + "subprocess timed out after " + str(timeout) + "s"
        )
        return status, detailed_status
    except subprocess.CalledProcessError:
        logging.error("subprocess called process error")
        status = LTLMCStatus("error")
        detailed_status = status.token().upper() + ":\n" + "subprocess called process error"
        return status, detailed_status
    except Exception as error:
        logging.critical(error)
        status = LTLMCStatus("error")
        detailed_status = status.token().upper() + ":\n" + "exception raised: " + str(error)
        return status, detailed_status
    out = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if out == "SUCCESS\n":
        return LTLMCStatus("satisfied"), ""
    if out == "FAILURE\n":
        return LTLMCStatus("violated"), ""
    if out == "ERROR: Inputs don't match\n":
        status = LTLMCStatus("invalid")
        detailed_status = status.token().upper() + ":\n" + "ERROR: Inputs don't match"
        return status, detailed_status
    if out == "ERROR: Outputs don't match\n":
        status = LTLMCStatus("invalid")
        detailed_status = status.token().upper() + ":\n" + "ERROR: Outputs don't match"
        return status, detailed_status
    if err.startswith("error: cannot read implementation file"):
        status = LTLMCStatus("invalid")
        detailed_status = status.token().upper() + ":\n" + "error: cannot read implementation file"
        return status, detailed_status
    logging.info("OUT: " + out)
    logging.info("ERR: " + err)
    status = LTLMCStatus("error")
    detailed_status = status.token().upper() + ":\n" + "OUT: " + out + "\nERR: " + err
    return status, detailed_status
