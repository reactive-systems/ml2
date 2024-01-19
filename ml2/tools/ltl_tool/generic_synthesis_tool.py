"""Generic synthesis tool"""

import logging
from datetime import timedelta
from typing import Generator

from grpc._channel import _InactiveRpcError

from ...grpc.ltl import ltl_syn_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_syn import LTLSynStatus
from ..grpc_service import GRPCService
from .tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericSynthesisTool(GRPCService):
    def __init__(self, setup_args=None, **kwargs):
        if setup_args is None:
            setup_args = {}
        super().__init__(stub=ltl_syn_pb2_grpc.GenericLTLSynthesisStub, **kwargs)
        setup_response = self.stub.Setup(SetupRequest(parameters=setup_args))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )

    def synthesize(self, problem: ToolLTLSynProblem) -> ToolLTLSynSolution:
        try:
            return ToolLTLSynSolution.from_pb2_LTLSynSolution(
                self.stub.Synthesize(problem.to_pb2_LTLSynProblem())
            )
        except _InactiveRpcError as err:
            return ToolLTLSynSolution(
                status=LTLSynStatus("error"),
                detailed_status="ERROR:\n" + str(err),
                tool="Generic Synthesis Tool",
                time=timedelta(0),
            )

    def synthesize_stream(
        self,
        problems: Generator[ToolLTLSynProblem, None, None],
    ) -> Generator[ToolLTLSynSolution, None, None]:
        for solution in self.stub.SynthesizeStream(problems):
            yield ToolLTLSynSolution.from_pb2_LTLSynSolution(solution)
