"""Limboole"""

import logging
from typing import Dict

from ...globals import CONTAINER_REGISTRY
from ...grpc.limboole import limboole_pb2_grpc
from ...grpc.prop import prop_pb2
from ...prop import PropFormula, PropSatStatus, PropValidStatus
from ..grpc_service import GRPCService
from .limboole_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMBOOLE_IMAGE_NAME = CONTAINER_REGISTRY + "/limboole-grpc-server:latest"


class Limboole(GRPCService):
    def __init__(
        self,
        image: str = LIMBOOLE_IMAGE_NAME,
        service=serve,
        cpu_count: int = 1,
        **kwargs,
    ):
        super().__init__(
            stub=limboole_pb2_grpc.LimbooleStub,
            image=image,
            cpu_count=cpu_count,
            tool="Limboole",
            service=service,
            **kwargs,
        )

    def check_sat(self, formula: PropFormula, timeout: int = None):
        # replace 0 (false) and 1 (true) in formula as they are not supported by Limboole
        formula_no_cons = " ".join(
            [
                "( a & ! a )" if t == "0" else "( a | ! a )" if t == "1" else t
                for t in formula.to_tokens()
            ]
        )
        pb_problem = prop_pb2.PropSatProblem(formula=formula_no_cons, timeout=timeout)
        pb_solution = self.stub.CheckSat(pb_problem)
        return PropSatStatus(pb_solution.status), pb_solution.assignment

    def check_solution(
        self, formula: PropFormula, assignment: Dict[str, int], timeout: int = None
    ):
        # TODO add method to PropFormula
        new_formula_str = formula.to_str("infix")
        for var, val in assignment.items():
            if val == 0:
                new_formula_str += f" & ! {var}"
            if val == 1:
                new_formula_str += f" & {var}"
        new_formula = PropFormula.from_str(new_formula_str)
        return self.check_sat(new_formula, timeout)[0]

    def check_valid(self, formula: PropFormula, timeout: int = None):
        # replace 0 (false) and 1 (true) in formula as they are not supported by Limboole
        formula_no_cons = " ".join(
            [
                "( a & ! a )" if t == "0" else "( a | ! a )" if t == "1" else t
                for t in formula.to_tokens()
            ]
        )
        pb_problem = prop_pb2.PropSatProblem(formula=formula_no_cons, timeout=timeout)
        pb_solution = self.stub.CheckValid(pb_problem)
        return PropValidStatus(pb_solution.status), pb_solution.assignment
