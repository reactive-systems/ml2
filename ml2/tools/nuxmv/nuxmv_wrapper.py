"""Wrapper for model checking script from Strix that uses the nuXmv model checker"""

import logging
import os
import subprocess

from ...ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ...ltl.ltl_spec import LTLSpec


def nuxmv_wrapper_dict(
    specification: dict,
    circuit: str,
    realizable: bool,
    strix_path,
    temp_dir,
    timeout: float = 10.0,
):
    spec_obj = LTLSpec.from_dict(specification)
    return nuxmv_wrapper(spec_obj, circuit, realizable, strix_path, temp_dir, timeout)


def nuxmv_wrapper(
    specification, circuit: str, realizable: bool, strix_path, temp_dir, timeout: float = 10.0
):
    temp_dir = os.path.join(temp_dir, "nuxmv")
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    specification.to_file(temp_dir, "specification.tlsf", format="tlsf")
    circuit_filepath = os.path.join(temp_dir, "circuit.aag")
    with open(circuit_filepath, "w") as aiger_file:
        aiger_file.write(circuit)
    verfiy_script_path = os.path.join(strix_path, "scripts/verify.sh")
    try:
        args = [verfiy_script_path, circuit_filepath, os.path.join(temp_dir, "specification.tlsf")]
        if realizable:
            args.append("REALIZABLE")
        else:
            args.append("UNREALIZABLE")
        args.append(str(timeout))
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.debug("subprocess timeout")
        return LTLMCStatus.TIMEOUT
    except subprocess.CalledProcessError:
        logging.error("subprocess called process error")
        return LTLMCStatus.ERROR
    except Exception as error:
        logging.critical(error)
    out = result.stdout.decode("utf-8")
    err = result.stderr.decode("utf-8")
    if out == "SUCCESS\n":
        return LTLMCStatus.SATISFIED
    if out == "FAILURE\n":
        return LTLMCStatus.VIOLATED
    if err.startswith("error: cannot read implementation file"):
        return LTLMCStatus.INVALID
    print(out)
    print(err)
    return LTLMCStatus.ERROR
