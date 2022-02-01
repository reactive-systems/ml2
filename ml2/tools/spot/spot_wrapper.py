"""Spot wrapper"""

import signal
import spot

from ...ltl.ltl_sat.ltl_sat_status import LTLSatStatus
from ...trace.trace_mc_status import TraceMCStatus


class TimeoutException(Exception):
    pass


def raise_timeout_exception(signum, frame):
    raise TimeoutException()


def automaton_trace(formula: str, simplify: bool = False, timeout: int = None):

    if timeout:
        signal.signal(signal.SIGALRM, raise_timeout_exception)
        signal.alarm(timeout)

    try:
        spot_formula = spot.formula(formula)
        automaton = spot_formula.translate()
        automaton.merge_edges()
        acc_run = automaton.accepting_run()
    except TimeoutException:
        return {"status": LTLSatStatus.TIMEOUT}

    signal.alarm(0)

    if acc_run is None:
        return {"status": LTLSatStatus.UNSATISFIABLE}
    else:
        trace = spot.twa_word(acc_run)

        if simplify:
            trace.simplify()

        return {"status": LTLSatStatus.SATISFIABLE, "trace": str(trace)}


def mc_trace(formula: str, trace: str, timeout: int = None):

    if timeout:
        signal.signal(signal.SIGALRM, raise_timeout_exception)
        signal.alarm(timeout)

    try:
        spot_formula = spot.formula(formula)
        spot_trace = spot.parse_word(trace)
        formula_aut = spot_formula.translate()
        trace_aut = spot_trace.as_automaton()

        if spot.contains(formula_aut, trace_aut):
            signal.alarm(0)
            return TraceMCStatus.SATISFIED
        else:
            signal.alarm(0)
            return TraceMCStatus.VIOLATED

    except TimeoutException:
        return TraceMCStatus.TIMEOUT
