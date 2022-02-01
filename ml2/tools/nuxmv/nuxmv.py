"""nuXmv"""

import logging

from ...globals import CONTAINER_REGISTRY
from ...ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import nuxmv_pb2, nuxmv_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUXMV_IMAGE_NAME = CONTAINER_REGISTRY + "/nuxmv-grpc-server:latest"


class nuXmv(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 2, mem_limit: str = "2g", port: int = None):
        super().__init__(NUXMV_IMAGE_NAME, cpu_count, mem_limit, port, service_name="nuXmv")
        self.stub = nuxmv_pb2_grpc.nuXmvStub(self.channel)

    def model_check(self, spec, system: str, realizable: bool = True, timeout: float = 10.0):
        specification = ltl_pb2.LTLSpecification(
            inputs=spec.inputs,
            outputs=spec.outputs,
            guarantees=spec.guarantees,
            assumptions=spec.assumptions,
        )
        pb_problem = nuxmv_pb2.Problem(
            specification=specification, system=system, realizable=realizable, timeout=timeout
        )
        pb_solution = self.stub.ModelCheck(pb_problem)
        return LTLMCStatus(nuxmv_pb2.Solution.Status.Name(pb_solution.status).lower())
