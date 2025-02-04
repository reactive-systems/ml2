"""Spot wrapper"""

import logging
import signal
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from multiprocessing import Manager, Process
from typing import Any, Dict, Generator, Optional, Set, Tuple

import spot

from ml2.dtypes.binary_ast import BinaryAST

from ...ltl.ltl_formula import LTLFormula
from ...ltl.ltl_mc import LTLMCStatus
from ...ltl.ltl_sat import LTLSatStatus
from ...ltl.ltl_syn import LTLSynStatus
from ...trace import TraceMCStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LTLSYNT_BIN_PATH = "ltlsynt"


class TimeoutException(Exception):
    pass


def raise_timeout_exception(signum, frame):
    raise TimeoutException()


def ltlsynt(
    formula_str, ins_str, outs_str, parameters: Dict[str, str], timeout=None, system_format="aiger"
) -> Dict[str, Any]:
    try:
        args = [LTLSYNT_BIN_PATH, "-f", formula_str]
        if ins_str:
            args.append(f"--ins={ins_str}")
        if outs_str:
            args.append(f"--outs={outs_str}")
        if system_format == "aiger":
            args.append("--aiger")
        for k, i in parameters.items():
            args.append(k)
            if i != "":
                args.append(i)
        logger.debug("subprocess args: %s", args)
        out = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("LTLSynt timeout")
        return {"status": LTLSynStatus("timeout")}
    except subprocess.CalledProcessError as e:
        logger.error("subprocess called process error")
        ltlsynt_stdout = e.stdout.decode("utf-8")
        ltlsynt_stderr = e.stderr.decode("utf-8")
        return {
            "status": LTLSynStatus("error"),
            "message": "OUT:\n" + ltlsynt_stdout + "ERR:\n" + ltlsynt_stderr,
        }
    except Exception as error:
        logger.critical(error)
        return {"status": LTLSynStatus("error"), "message": str(error)}
    logger.debug("LTLSynt returncode: %s", out.returncode)
    ltlsynt_stdout = out.stdout.decode("utf-8")
    logger.debug("LTLSynt stdout: %s", ltlsynt_stdout)
    ltlsynt_stdout_lines = ltlsynt_stdout.splitlines()
    if (
        "Warning: The current network has no primary outputs. Some commands may not work correctly."
        in ltlsynt_stdout_lines
    ):
        return {"status": LTLSynStatus("error"), "message": ltlsynt_stdout}
    if out.returncode == 0 and ltlsynt_stdout_lines[0] == "REALIZABLE":
        logger.debug("realizable")
        system = "\n".join(ltlsynt_stdout_lines[1:])
        if system_format == "aiger":
            logger.debug("AIGER circuit: %s", system)
            return {"status": LTLSynStatus("realizable"), "circuit": system}
        elif system_format == "mealy":
            logger.debug("Mealy Machine: %s", system)
            return {"status": LTLSynStatus("realizable"), "mealy_machine": system}
    if out.returncode == 0 and ltlsynt_stdout_lines[0] == "UNREALIZABLE":
        logger.debug("unrealizable")
        system = "\n".join(ltlsynt_stdout_lines[1:])
        if system_format == "aiger":
            logger.debug("AIGER circuit: %s", system)
            return {"status": LTLSynStatus("unrealizable"), "circuit": system}
        elif system_format == "mealy":
            logger.debug("Mealy Machine: %s", system)
            return {"status": LTLSynStatus("unrealizable"), "mealy_machine": system}
    logger.debug("LTLSynt error")
    return {"status": LTLSynStatus("error"), "message": "OUT:\n" + ltlsynt_stdout}


def check_sat(formula: str, simplify: bool = False, timeout: int = None):
    start = time.time()
    if timeout:
        manager = Manager()
        result = manager.dict()
        process = Process(target=check_sat_with_signal, args=(formula, result, simplify, timeout))
        process.start()
        process.join(timeout)
        process.terminate()
        end = time.time()
        logger.info("Multiprocessing and checking satisfiability took %f seconds", end - start)
        if process.exitcode == 0:
            return {
                "status": result["status"],
                "trace": result["trace"] if "trace" in result else None,
            }
        elif process.exitcode is None:
            return {"status": LTLSatStatus("timeout")}
        else:
            return {"status": LTLSatStatus("error")}
    else:
        result = {}
        check_sat_with_signal(formula, result, simplify)
        end = time.time()
        logger.info("Checking satisfiability took %f seconds", end - start)
        return result


def check_sat_with_signal(formula: str, result: dict, simplify: bool = False, timeout: int = None):
    if timeout:
        signal.signal(signal.SIGALRM, raise_timeout_exception)
        signal.alarm(timeout)

    try:
        spot_formula = spot.formula(formula)
        automaton = spot_formula.translate()
        automaton.merge_edges()
        acc_run = automaton.accepting_run()
    except TimeoutException:
        result["status"] = LTLSatStatus("timeout")

    signal.alarm(0)

    if acc_run is None:
        result["status"] = LTLSatStatus("unsatisfiable")
    else:
        trace = spot.twa_word(acc_run)

        if simplify:
            trace.simplify()

        result["status"] = LTLSatStatus("satisfiable")
        result["trace"] = str(trace)


def check_equiv_renaming(formula1: str, formula2: str, result: dict) -> None:
    NO_CHANGE_TOKEN = "NO_CHANGE_TOKEN"

    start = time.time()

    try:

        def parse(formula: str) -> BinaryAST:
            return LTLFormula.parse("{0:fp}".format(spot.formula(formula)))

        f1 = parse(formula1)
        f2 = parse(formula2)

        def aps(formula: BinaryAST) -> Set[str]:
            return set(x for x in formula.leaves if x not in ["1", "0"])  # for true and false

        if len(aps(f1)) > len(aps(f2)):
            key_formula = f1
            value_formula = f2
        else:
            key_formula = f2
            value_formula = f1

        key_aps = aps(key_formula)
        value_aps = aps(value_formula)

        renames = [
            list(zip(key_aps, item))
            for item in product(value_aps.union({NO_CHANGE_TOKEN}), repeat=len(key_aps))
        ]

        renames = list(
            filter(
                lambda l: (
                    len(set(a[1] for a in l).difference([NO_CHANGE_TOKEN]))
                    + [a[1] for a in l].count(NO_CHANGE_TOKEN)
                    == len(key_aps)
                )
                and (len(set(a[1] for a in l).difference([NO_CHANGE_TOKEN])) == len(value_aps)),
                renames,
            )
        )

        renames = [{x: (y if y != NO_CHANGE_TOKEN else x) for x, y in el} for el in renames]

        def rename(f, d):
            f = deepcopy(f)
            f.rename(d)
            return f

        spot_f1s = [spot.formula(rename(key_formula, d).to_str(notation="infix")) for d in renames]

        spot_f2 = spot.formula(value_formula.to_str(notation="infix"))

        lcc = spot.language_containment_checker()

        if any(lcc.are_equivalent(spot_f2, spot_f1) for spot_f1 in spot_f1s):
            end = time.time()
            result["status"] = "equivalent"
            result["time"] = end - start
        else:
            end = time.time()
            result["status"] = "inequivalent"
            result["time"] = end - start
    except Exception as e:
        print(e)
        result["status"] = "error"
        end = time.time()
        result["time"] = end - start


# def check_equiv(formula1: str, formula2: str, result: dict) -> None:
#     start = time.time()

#     spot_formula_1 = spot.formula(formula1)
#     spot_formula_2 = spot.formula(formula2)

#     try:
#         if spot.are_equivalent(spot_formula_1, spot_formula_2):
#             end = time.time()
#             result["status"] = "equivalent"
#             result["time"] = end - start
#         else:
#             end = time.time()
#             result["status"] = "inequivalent"
#             result["time"] = end - start
#     except Exception as e:
#         print(e)
#         result["status"] = "error"
#         end = time.time()
#         result["time"] = end - start


def check_equiv(formula1: str, formula2: str, result: dict) -> None:
    start = time.time()

    spot_formula_1 = spot.formula(formula1)
    spot_formula_2 = spot.formula(formula2)

    try:
        aut_1 = spot_formula_1.translate()
        aut_2 = spot_formula_2.translate()
        trace = aut_1.exclusive_word(aut_2)
        end = end = time.time()
        if trace is None:
            result["status"] = "equivalent"
        else:
            result["status"] = "inequivalent"
            result["exclusive_word"] = str(trace)
        result["time"] = end - start
    except Exception as e:
        print(e)
        result["status"] = "error"
        end = time.time()
        result["time"] = end - start


def inclusion(
    problems: Generator[Tuple[str, str], None, None]
) -> Generator[Tuple[str, float], None, None]:

    lcc = spot.language_containment_checker()
    for left, right in problems:
        start = time.time()

        try:
            spot_left = spot.formula(left)
            spot_right = spot.formula(right)

            if lcc.are_equivalent(spot_left, spot_right):
                end = time.time()
                yield "equivalent", end - start

            elif spot_left.contains(spot_right):
                end = time.time()
                yield "only_left_in_right", end - start
            elif spot_right.contains(spot_left):
                end = time.time()
                yield "only_right_in_left", end - start
            else:
                end = time.time()
                yield "incomparable", end - start
        except Exception as e:
            print(e)
            end = time.time()
            yield "error", end - start


def mc_trace(formula: str, trace: str, result):
    try:
        spot_formula = spot.formula(formula)
        spot_trace = spot.parse_word(trace)
        formula_aut = spot_formula.translate()
        trace_aut = spot_trace.as_automaton()

        if spot.contains(formula_aut, trace_aut):
            signal.alarm(0)
            result["status"] = TraceMCStatus("satisfied")
        else:
            signal.alarm(0)
            result["status"] = TraceMCStatus("violated")

    except TimeoutException:
        result["status"] = TraceMCStatus("timeout")


def model_check_system(
    formula: str,
    circuit: Optional[str],
    mealy_machine: Optional[str],
    realizable: bool,
    result: dict,
) -> None:
    start = datetime.now()
    f = spot.formula(formula)
    if realizable:
        f_neg = spot.formula_Not(f)
    else:
        f_neg = spot.formula(f)
    aut_f_neg = spot.translate(f_neg)
    if circuit is not None:
        aiger = spot.aiger_circuit(circuit)
        aut = spot.automaton(aiger.as_automaton().to_str())  # Very weird hack
    elif mealy_machine is not None:
        aut = spot.automaton(mealy_machine)
    counterexample = aut.intersecting_word(aut_f_neg)
    duration = datetime.now() - start
    if counterexample:
        result["counterexample"] = str(counterexample)
        result["status"] = LTLMCStatus("violated")
        result["time"] = duration
    else:
        result["counterexample"] = None
        result["status"] = LTLMCStatus("satisfied")
        result["time"] = duration
    return
