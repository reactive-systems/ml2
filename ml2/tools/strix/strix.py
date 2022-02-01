"""Strix"""

import logging

from ...globals import CONTAINER_REGISTRY
from ...ltl.ltl_syn import LTLSynStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import strix_pb2, strix_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRIX_IMAGE_NAME = CONTAINER_REGISTRY + "/strix-grpc-server:latest"


class Strix(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 2, mem_limit: str = "2g", port: int = None):
        super().__init__(STRIX_IMAGE_NAME, cpu_count, mem_limit, port, service_name="Strix")
        self.stub = strix_pb2_grpc.StrixStub(self.channel)

    def synthesize(
        self, spec, minimize_aiger=False, minimize_mealy=False, threads=None, timeout=None
    ):
        pb_spec = ltl_pb2.LTLSpecification(
            inputs=spec.inputs,
            outputs=spec.outputs,
            guarantees=spec.guarantees,
            assumptions=spec.assumptions,
        )
        pb_problem = strix_pb2.StrixProblem(
            specification=pb_spec,
            minimize_aiger=minimize_aiger,
            minimize_mealy=minimize_mealy,
            threads=threads,
            timeout=timeout,
        )
        pb_solution = self.stub.Synthesize(pb_problem)
        return LTLSynStatus(pb_solution.status), pb_solution.system

    def synthesize_data(
        self, data, minimize_aiger=False, minimize_mealy=False, threads=None, timeout=None
    ):
        stats = {}
        for spec in data.dataset:
            status, _ = self.synthesize(spec, minimize_aiger, minimize_mealy, threads, timeout)
            stats[status] = stats.get(status, 0) + 1
            print(stats)
        return stats
