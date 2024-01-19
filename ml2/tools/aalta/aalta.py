"""Aalta"""

import logging
from functools import reduce

from ...globals import CONTAINER_REGISTRY
from ...grpc.aalta import aalta_pb2, aalta_pb2_grpc
from ...ltl import LTLFormula
from ...ltl.ltl_sat import LTLSatStatus
from ...registry import register_type
from ...trace import SymbolicTrace, Trace, TraceMCStatus
from ..grpc_service import GRPCService
from .aalta_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AALTA_IMAGE_NAME = CONTAINER_REGISTRY + "/aalta-grpc-server:latest"


class Aalta(GRPCService):
    def __init__(
        self,
        image: str = AALTA_IMAGE_NAME,
        cpu_count: int = 1,
        service=serve,
        tool="Aalta",
        **kwargs,
    ):
        super().__init__(
            stub=aalta_pb2_grpc.AaltaStub,
            image=image,
            cpu_count=cpu_count,
            tool=tool,
            service=service,
            **kwargs,
        )

    def check_sat(self, formula: LTLFormula, timeout: int = None):
        pb_problem = aalta_pb2.LTLSatProblemAalta(
            formula=formula.to_str(notation="infix"), timeout=timeout
        )
        pb_solution = self.stub.CheckSat(pb_problem)
        trace = Trace.from_aalta_str(pb_solution.trace).to_str() if pb_solution.trace else None
        return LTLSatStatus(pb_solution.status), trace


def encode_for_satisfiability(trace: SymbolicTrace, formula: LTLFormula):
    # prefix
    step_constraints = []
    for idx, trace_step_formula in enumerate(trace.prefix):
        for _ in range(idx):  # prepend X's for step
            trace_step_formula = "X " + trace_step_formula
        step_constraints.append(trace_step_formula)
    prefix_part = ""
    if step_constraints:
        prefix_part = reduce(lambda x, y: f"& {x} {y}", step_constraints)  # AND together

    # generate encoding aps for cycle steps
    cycle_encoding_bits = bin(len(trace.cycle))[2:]
    # used_aps = trace_obj.contained_aps() | formula.contained_aps() # TODO remove?
    num_encoding_aps = len(cycle_encoding_bits)
    encoding_aps = ["c" + str(q) for q in range(num_encoding_aps)]

    # build encodings for cycle steps
    encodings = []
    for idx, _ in enumerate(trace.cycle):
        bin_rep = "{{:0{:d}b}}".format(num_encoding_aps).format(idx)
        encoding = []
        for idx_encode, c in enumerate(bin_rep):
            ap = encoding_aps[idx_encode]
            if c == "1":
                encoding.append(ap)
            elif c == "0":
                encoding.append("! " + ap)
            else:
                raise ValueError()
        encodings.append(reduce(lambda x, y: f"& {x} {y}", encoding))

    # build "chain" between cycle steps
    cycle_constraints = []
    for idx, _ in enumerate(trace.cycle):
        if idx + 1 == len(trace.cycle):  # last step in cycle
            next_idx = 0
        else:
            next_idx = idx + 1
        cycle_constraints.append(
            f"G -> {encodings[idx]} X & {encodings[next_idx]} {trace.cycle[next_idx]}"
        )
    cycle_part = reduce(
        lambda x, y: f"& {x} {y}", cycle_constraints
    )  # and step formulas together, add to complete formula

    # start chain
    cycle_part = f"& & {cycle_part} {encodings[0]} {trace.cycle[0]}"

    # prepend nexts to cycle
    for _ in range(len(trace.prefix)):
        cycle_part = f"X {cycle_part}"

    # add Nexts to cycle part, add formula to check
    if prefix_part:
        complete = f"& & {prefix_part} {cycle_part} ! {formula.to_str()}"
    else:
        complete = f"& {cycle_part} ! {formula.to_str()}"
    return complete


@register_type
class AaltaTraceMC(Aalta):
    def verify(
        self,
        problem: LTLFormula,
        solution: SymbolicTrace,
        timeout: int = 120,
        **kwargs,
    ) -> TraceMCStatus:
        assert problem._notation == "prefix"
        assert solution._notation == "prefix"
        sat_encoding = encode_for_satisfiability(solution, problem)
        status, trace = self.check_sat(
            formula=LTLFormula.from_str(sat_encoding, notation="prefix"),
            timeout=timeout,
        )
        if status == LTLSatStatus("unsatisfiable"):
            return TraceMCStatus("satisfied")
        elif status == LTLSatStatus("satisfiable"):
            return TraceMCStatus("violated")
        elif status == LTLSatStatus("timeout"):
            return TraceMCStatus("timeout")
        else:
            return TraceMCStatus("error")
