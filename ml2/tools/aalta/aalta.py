"""Aalta"""

import logging

from ...globals import CONTAINER_REGISTRY
from ...ltl.ltl_sat import LTLSatStatus
from ...trace import Trace
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import aalta_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AALTA_IMAGE_NAME = CONTAINER_REGISTRY + "/aalta-grpc-server:latest"


class Aalta(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 1, mem_limit: str = "2g", port: int = None):
        super().__init__(AALTA_IMAGE_NAME, cpu_count, mem_limit, port, service_name="Aalta")
        self.stub = aalta_pb2_grpc.AaltaStub(self.channel)

    def check_sat(self, formula: str, timeout: int = None):
        pb_problem = ltl_pb2.LTLSatProblem(formula=formula, timeout=timeout)
        pb_solution = self.stub.CheckSat(pb_problem)
        trace = Trace.from_aalta_str(pb_solution.trace).to_str() if pb_solution.trace else None
        return LTLSatStatus(pb_solution.status), trace
