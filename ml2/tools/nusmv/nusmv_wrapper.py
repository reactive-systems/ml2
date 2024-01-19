"""Wrapper for model checking using the NuSMV model checker"""

import logging
import os
import re
import subprocess
from typing import Any, Dict, Optional

from ...aiger.aiger_circuit import AIGERCircuit
from ...ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ...ltl.ltl_spec.ltl_spec import LTLSpec
from ...trace import Trace

SATISFIED_PATTERN = re.compile(r"^-- specification (.*) is true$", re.MULTILINE)
VIOLATED_PATTERN = re.compile(r"^-- specification (.*) is false$", re.MULTILINE)
COUNTER_PATTERN = re.compile(
    r".*Trace Description: LTL Counterexample \nTrace Type: Counterexample \n(.*)", re.DOTALL
)


# TODO aig_to_smv and ltlfilt timeout


def nusmv_wrapper(
    spec: LTLSpec,
    circuit: AIGERCircuit,
    realizable: bool,
    aig_to_smv_path: str,
    ltlfilt_path: str,
    nusmv_path: str,
    temp_dir: str,
    timeout: Optional[float] = 10.0,
) -> Dict[str, Any]:
    temp_dir = os.path.join(temp_dir, "nusmv")
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    try:
        # translate aiger to nusmv
        circuit_str = circuit.to_str()
        # aig_to_smv requires newline character at the end of the aiger file
        if not circuit_str.endswith("\n"):
            circuit_str = circuit_str + "\n"
        aiger_filepath = os.path.join(temp_dir, "circuit.aag")
        with open(aiger_filepath, "w") as aiger_file:
            aiger_file.write(circuit_str)
        aig_to_smv_args = [aig_to_smv_path, aiger_filepath]
        aig_to_smv_result = subprocess.run(aig_to_smv_args, capture_output=True)
        nusmv_circuit = aig_to_smv_result.stdout.decode("utf-8")

        # prepare specification
        ltlfilt_args = [ltlfilt_path, "-p", "--unabbreviate=WMR"]
        if not realizable:
            ltlfilt_args.append("--negate")
        ltlfilt_args.append("-f")
        ltlfilt_args.append(spec.to_str())
        ltlfilt_result = subprocess.run(ltlfilt_args, capture_output=True)
        ltlfilt_spec = ltlfilt_result.stdout.decode("utf-8")

        # model checking
        nusmv_input = f"{nusmv_circuit}LTLSPEC {ltlfilt_spec}"
        nusmv_input_filepath = os.path.join(temp_dir, "problem.smv")
        with open(nusmv_input_filepath, "w") as nusmv_file:
            nusmv_file.write(nusmv_input)
        nusmv_result = subprocess.run(
            [nusmv_path, nusmv_input_filepath], capture_output=True, timeout=timeout
        )
        nusmv_out = nusmv_result.stdout.decode("utf-8")

    except subprocess.TimeoutExpired:
        logging.debug("subprocess timeout")
        return {"status": LTLMCStatus("timeout")}
    except subprocess.CalledProcessError:
        logging.error("subprocess called process error")
        return {"status": LTLMCStatus("error")}
    except Exception as error:
        logging.critical(error)
        return {"status": LTLMCStatus("error"), "detailed_status": str(error)}

    if SATISFIED_PATTERN.search(nusmv_out):
        return {"status": LTLMCStatus("satisfied"), "detailed_status": nusmv_out}
    if VIOLATED_PATTERN.search(nusmv_out):
        if m := COUNTER_PATTERN.match(nusmv_out):
            counterexample = Trace.from_str(m.group(1), notation="nusmv")
        else:
            counterexample = None
        return {
            "status": LTLMCStatus("violated"),
            "detailed_status": nusmv_out,
            "counterexample": counterexample,
        }
    return {"status": LTLMCStatus("error"), "detailed_status": nusmv_out}
