"""NuSMV"""

import logging
from datetime import timedelta
from typing import Any, Dict, Generator, Optional

from grpc._channel import _InactiveRpcError

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq
from ...globals import CONTAINER_REGISTRY
from ...grpc.nusmv import nusmv_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_mc import LTLMCSolution, LTLMCStatus
from ...ltl.ltl_spec import DecompLTLSpec
from ...ltl.ltl_syn import LTLRealStatus
from ...verifier.verifier import Verifier
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_mc_problem import ToolLTLMCProblem, ToolLTLMCSolution
from .nusmv_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUSMV_IMAGE_NAME = CONTAINER_REGISTRY + "/nusmv-grpc-server:latest"


class NuSMV(GRPCService):
    def __init__(
        self,
        image: str = NUSMV_IMAGE_NAME,
        mem_limit="2g",
        service=serve,
        tool: str = "NuSMV",
        **kwargs
    ):
        super().__init__(
            image=image,
            mem_limit=mem_limit,
            service=service,
            stub=nusmv_pb2_grpc.NuSMVStub,
            tool=tool,
            **kwargs
        )
        setup_response = self.stub.Setup(SetupRequest(parameters={}))
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
                tool="NuSMV",
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


class NuSMVMC(NuSMV, Verifier):
    def verify(
        self,
        formula: DecompLTLSpec,
        solution: CatSeq[LTLRealStatus, AIGERCircuit],
        target=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> LTLMCSolution:
        if parameters is None:
            parameters = {}
        return self.model_check(
            problem=ToolLTLMCProblem.from_aiger_verification_pair(
                formula=formula, solution=solution, parameters=parameters
            )
        ).to_LTLMCSolution()
