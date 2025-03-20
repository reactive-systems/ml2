"""SemML"""

import logging
from datetime import timedelta
from typing import Generator

from grpc._channel import _InactiveRpcError

from ...globals import CONTAINER_REGISTRY
from ...grpc.semml import semml_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_syn import LTLSynStatus
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from .semml_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEMML_IMAGE_NAME = CONTAINER_REGISTRY + "/semml-grpc-server:latest"


class Semml(GRPCService):
    def __init__(self, image: str = SEMML_IMAGE_NAME, service=serve, **kwargs):
        super().__init__(
            stub=semml_pb2_grpc.SemmlStub, image=image, service=service, tool="SemML", **kwargs
        )
        setup_response = self.stub.Setup(SetupRequest(parameters={}))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )

    def synthesize(self, problem: ToolLTLSynProblem) -> ToolLTLSynSolution:
        try:
            if problem.system_format != "aiger":
                raise NotImplementedError("SemML can only synthesize AIGER for now.")
            return ToolLTLSynSolution.from_pb2_LTLSynSolution(
                self.stub.Synthesize(problem.to_pb2_LTLSynProblem())
            )
        except _InactiveRpcError as err:
            return ToolLTLSynSolution(
                status=LTLSynStatus("error"),
                detailed_status="ERROR:\n" + str(err),
                tool="SemML",
                time=timedelta(0),
            )

    def synthesize_stream(
        self,
        problems: Generator[ToolLTLSynProblem, None, None],
    ) -> Generator[ToolLTLSynSolution, None, None]:
        def _problems(problems):
            for problem in problems:
                if problem.system_format != "aiger":
                    raise NotImplementedError("SemML can only synthesize AIGER for now.")
                yield problem.to_pb2_LTLSynProblem()

        for solution in self.stub.SynthesizeStream(_problems(problems)):
            yield ToolLTLSynSolution.from_pb2_LTLSynSolution(solution)
