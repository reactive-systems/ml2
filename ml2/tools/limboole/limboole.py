"""Limboole"""

import logging
from typing import Dict

from ml2.prop.prop_formula import PropFormula

from ...globals import CONTAINER_REGISTRY
from ...prop import PropSatStatus, PropValidStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import prop_pb2
from . import limboole_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMBOOLE_IMAGE_NAME = CONTAINER_REGISTRY + "/limboole-grpc-server:latest"


class Limboole(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 1, mem_limit: str = "2g", port: int = None):
        super().__init__(LIMBOOLE_IMAGE_NAME, cpu_count, mem_limit, port, service_name="Limboole")
        self.stub = limboole_pb2_grpc.LimbooleStub(self.channel)

    def check_sat(self, formula: str, timeout: int = None):
        # replace 0 (false) and 1 (true) in formula as they are not supported by Limboole
        formula_no_cons = " ".join(
            [
                "( a & ! a )" if t == "0" else "( a | ! a )" if t == "1" else t
                for t in PropFormula.from_str(formula).tokens()
            ]
        )
        pb_problem = prop_pb2.PropSatProblem(formula=formula_no_cons, timeout=timeout)
        pb_solution = self.stub.CheckSat(pb_problem)
        return PropSatStatus(pb_solution.status), pb_solution.assignment

    def check_solution(self, formula: str, assignment: Dict[str, int], timeout: int = None):
        for var, val in assignment.items():
            if val == 0:
                formula += f" & ! {var}"
            if val == 1:
                formula += f" & {var}"
        return self.check_sat(formula, timeout)[0]

    def check_valid(self, formula: str, timeout: int = None):
        # replace 0 (false) and 1 (true) in formula as they are not supported by Limboole
        formula_no_cons = " ".join(
            [
                "( a & ! a )" if t == "0" else "( a | ! a )" if t == "1" else t
                for t in PropFormula.from_str(formula).tokens()
            ]
        )
        pb_problem = prop_pb2.PropSatProblem(formula=formula_no_cons, timeout=timeout)
        pb_solution = self.stub.CheckValid(pb_problem)
        return PropValidStatus(pb_solution.status), pb_solution.assignment
