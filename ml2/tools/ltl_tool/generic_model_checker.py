"""Generic Model Checker Client Implementation"""

import logging
from datetime import timedelta
from typing import Dict, Generator, Optional

from grpc._channel import _InactiveRpcError

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq
from ...grpc.ltl import ltl_mc_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_mc import LTLMCSolution, LTLMCStatus
from ...ltl.ltl_spec import DecompLTLSpec
from ...ltl.ltl_syn import LTLRealStatus
from ...verifier.verifier import Verifier
from ..grpc_service import GRPCService
from .tool_ltl_mc_problem import ToolLTLMCProblem, ToolLTLMCSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericModelCheckerTool(GRPCService):
    def __init__(self, setup_args=None, **kwargs):
        if setup_args is None:
            setup_args = {}
        super().__init__(stub=ltl_mc_pb2_grpc.GenericLTLModelCheckerStub, **kwargs)
        setup_response = self.stub.Setup(SetupRequest(parameters=setup_args))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )

    def model_check(
        self,
        problem: ToolLTLMCProblem,
    ) -> ToolLTLMCSolution:
        try:
            assert problem.circuit is not None
            return ToolLTLMCSolution.from_pb2_LTLMCSolution(
                self.stub.ModelCheck(problem.to_pb2_LTLMCProblem())
            )
        except _InactiveRpcError as err:
            return ToolLTLMCSolution(
                status=LTLMCStatus("error"),
                detailed_status="ERROR:\n" + str(err),
                tool="Generic Model Checker",
                time=timedelta(0),
            )

    def model_check_stream(
        self,
        problems: Generator[ToolLTLMCProblem, None, None],
    ) -> Generator[ToolLTLMCSolution, None, None]:
        def _problems(problems):
            for problem in problems:
                assert problem.circuit is not None
                yield problem.to_pb2_LTLMCProblem()

        for solution in self.stub.ModelCheckStream(_problems(problems)):
            yield ToolLTLMCSolution.from_pb2_LTLMCSolution(solution)


class GenericModelChecker(GenericModelCheckerTool, Verifier):
    def verify(
        self,
        formula: DecompLTLSpec,
        solution: CatSeq[LTLRealStatus, AIGERCircuit],
        parameters: Optional[Dict[str, str]] = None,
    ) -> LTLMCSolution:
        if parameters is None:
            parameters = {}
        return self.model_check(
            problem=ToolLTLMCProblem.from_aiger_verification_pair(
                formula=formula, solution=solution, parameters=parameters
            )
        ).to_LTLMCSolution()
