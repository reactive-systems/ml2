"""BoSy"""

import logging

from ...globals import CONTAINER_REGISTRY
from ...ltl import LTLSpec
from ...ltl.ltl_syn.ltl_syn_status import LTLSynStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import bosy_pb2, bosy_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOSY_IMAGE_NAME = CONTAINER_REGISTRY + "/bosy-grpc-server:latest"


class BoSy(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 1, mem_limit: str = "2g", port: int = None):
        super().__init__(BOSY_IMAGE_NAME, cpu_count, mem_limit, port, service_name="BoSy")
        self.stub = bosy_pb2_grpc.BoSyStub(self.channel)
        logger.info("Compiling BoSy ...")
        spec = LTLSpec.from_dict(
            {"guarantees": ["G (i -> F o)"], "inputs": ["i"], "outputs": ["o"]}
        )
        self.synthesize(spec)
        logger.info("Compiled Bosy")

    def synthesize(self, spec, optimize=False, timeout=None):
        pb_spec = ltl_pb2.LTLSpecification(
            inputs=spec.inputs,
            outputs=spec.outputs,
            guarantees=spec.guarantees,
            assumptions=spec.assumptions,
        )
        pb_problem = bosy_pb2.BoSyProblem(
            specification=pb_spec, optimize=optimize, timeout=timeout
        )
        pb_solution = self.stub.Synthesize(pb_problem)
        return LTLSynStatus(pb_solution.status), pb_solution.system
