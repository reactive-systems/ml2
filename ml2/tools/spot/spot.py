"""Spot"""

import logging
from typing import Iterator, List

from ...globals import CONTAINER_REGISTRY
from ...ltl.ltl_sat import LTLSatStatus
from ...trace import TraceMCStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import spot_pb2, spot_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPOT_IMAGE_NAME = CONTAINER_REGISTRY + "/spot-grpc-server:latest"


class Spot(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 1, mem_limit: str = "2g", port: int = None):
        super().__init__(SPOT_IMAGE_NAME, cpu_count, mem_limit, port, service_name="Spot")
        self.stub = spot_pb2_grpc.SpotStub(self.channel)

    def check_sat(self, formula: str, simplify: bool = False, timeout: int = None):
        pb_problem = ltl_pb2.LTLSatProblem(formula=formula, simplify=simplify, timeout=timeout)
        pb_solution = self.stub.CheckSat(pb_problem)
        return LTLSatStatus(pb_solution.status), pb_solution.trace

    def mc_trace(self, formula: str, trace: str, timeout: int = None):
        pb_problem = ltl_pb2.LTLTraceMCProblem(formula=formula, trace=trace, timeout=timeout)
        pb_solution = self.stub.MCTrace(pb_problem)
        return TraceMCStatus(pb_solution.status)

    def randltl(
        self,
        num_formulas: int = 1,
        num_aps: int = None,
        aps: List[str] = None,
        allow_dups: bool = False,
        output: str = "ltl",
        seed: int = 0,
        simplify: int = 0,
        tree_size: int = 15,
        boolean_priorities: str = None,
        ltl_priorities: str = None,
        sere_priorities: str = None,
    ) -> Iterator[str]:
        pb_args = spot_pb2.RandLTLArgs(
            num_formulas=num_formulas,
            num_aps=num_aps,
            aps=aps,
            allow_dups=allow_dups,
            output=output,
            seed=seed,
            simplify=simplify,
            tree_size=tree_size,
            boolean_priorities=boolean_priorities,
            ltl_priorities=ltl_priorities,
            sere_priorities=sere_priorities,
        )
        for pb_formula in self.stub.RandLTL(pb_args):
            yield pb_formula.formula
