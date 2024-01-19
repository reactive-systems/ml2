"""nuXmv"""

import logging
from datetime import timedelta
from typing import Dict, Generator, Optional

from grpc._channel import _InactiveRpcError

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq
from ...globals import CONTAINER_REGISTRY
from ...grpc.nuxmv import nuxmv_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_mc import LTLMCSolution, LTLMCStatus
from ...ltl.ltl_spec import LTLSpec
from ...ltl.ltl_syn import LTLRealStatus
from ...registry import register_type
from ...utils.dist_utils import architecture_is_apple_arm
from ...verifier.verifier import Verifier
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_mc_problem import ToolLTLMCProblem, ToolLTLMCSolution
from .nuxmv_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUXMV_IMAGE_NAME = CONTAINER_REGISTRY + "/nuxmv-grpc-server:latest"


class Nuxmv(GRPCService):
    def __init__(
        self,
        image: str = NUXMV_IMAGE_NAME,
        mem_limit="2g",
        service=serve,
        tool: str = "nuXmv",
        **kwargs
    ):
        if architecture_is_apple_arm():
            mem_limit = "8g"
            logger.warning("Memory limit set to 8g for nuXmv container because auf M1 chip")
        super().__init__(
            image=image,
            mem_limit=mem_limit,
            service=service,
            stub=nuxmv_pb2_grpc.NuxmvStub,
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
                tool="nuXmv",
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


@register_type
class NuxmvMC(Nuxmv, Verifier):
    def verify(
        self,
        problem: LTLSpec,
        solution: CatSeq[LTLRealStatus, AIGERCircuit],
        parameters: Optional[Dict[str, str]] = None,
    ) -> LTLMCSolution:
        if parameters is None:
            parameters = {}
        return self.model_check(
            problem=ToolLTLMCProblem.from_aiger_verification_pair(
                formula=problem, solution=solution, parameters=parameters
            )
        ).to_LTLMCSolution()

    def verify_aiger(
        self,
        formula: LTLSpec,
        real_status: bool,
        circuit: AIGERCircuit,
        parameters: Optional[Dict[str, str]] = None,
    ) -> LTLMCSolution:
        return self.model_check(
            ToolLTLMCProblem(
                parameters=parameters,
                realizable=real_status,
                specification=formula,
                circuit=circuit,
            )
        ).to_LTLMCSolution()
