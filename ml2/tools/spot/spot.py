"""Spot"""

import json
import logging
from datetime import timedelta
from typing import Generator, Iterator, List

from grpc._channel import _InactiveRpcError

from ...globals import CONTAINER_REGISTRY
from ...grpc.aiger import aiger_pb2
from ...grpc.ltl import ltl_equiv_pb2, ltl_sat_pb2, ltl_trace_mc_pb2
from ...grpc.mealy import mealy_pb2
from ...grpc.spot import spot_pb2, spot_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl import LTLFormula
from ...ltl.ltl_equiv import LTLEquivStatus
from ...ltl.ltl_mc import LTLMCStatus
from ...ltl.ltl_sat import LTLSatStatus
from ...ltl.ltl_syn import LTLSynStatus
from ...trace import SymbolicTrace, TraceMCStatus
from ..grpc_service import GRPCService
from ..ltl_tool import ToolLTLMCProblem, ToolLTLMCSolution, ToolLTLSynProblem, ToolLTLSynSolution
from .spot_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPOT_IMAGE_NAME = CONTAINER_REGISTRY + "/spot-grpc-server:latest"


class Spot(GRPCService):
    def __init__(
        self,
        cpu_count: int = 1,
        image: str = SPOT_IMAGE_NAME,
        service=serve,
        tool: str = "Spot",
        **kwargs
    ):
        super().__init__(
            stub=spot_pb2_grpc.SpotStub,
            image=image,
            cpu_count=cpu_count,
            service=service,
            tool=tool,
            **kwargs
        )
        setup_response = self.stub.Setup(SetupRequest(parameters={}))
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
                tool="Spot",
                time=timedelta(0),
            )

    def model_check(
        self,
        problem: ToolLTLMCProblem,
    ) -> ToolLTLMCSolution:
        try:
            return ToolLTLMCSolution.from_pb2_LTLMCSolution(
                self.stub.ModelCheck(problem.to_pb2_LTLMCProblem())
            )
        except _InactiveRpcError as err:
            return ToolLTLMCSolution(
                status=LTLMCStatus("error"),
                detailed_status="ERROR:\n" + str(err),
                tool="Spot",
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

    def check_equiv(self, f: LTLFormula, g: LTLFormula, timeout: float = None) -> LTLEquivStatus:
        pb_problem = ltl_equiv_pb2.LTLEquivProblem(
            formula1=f.to_str(notation="infix"),
            formula2=g.to_str(notation="infix"),
            timeout=timeout,
        )
        pb_solution = self.stub.CheckEquiv(pb_problem)
        return LTLEquivStatus(pb_solution.status)

    def check_sat(self, formula: LTLFormula, simplify: bool = False, timeout: int = None):
        pb_problem = ltl_sat_pb2.LTLSatProblem(
            formula=formula.to_str(notation="infix"), simplify=simplify, timeout=timeout
        )
        pb_solution = self.stub.CheckSat(pb_problem)
        status = LTLSatStatus(pb_solution.status)
        trace = (
            SymbolicTrace.from_str(pb_solution.trace, notation="infix", spot=True)
            if pb_solution.trace
            else None
        )
        return (status, trace)

    def mc_trace(self, formula: LTLFormula, trace: SymbolicTrace, timeout: int = None):
        pb_problem = ltl_trace_mc_pb2.LTLTraceMCProblem(
            formula=formula.to_str(notation="infix"),
            trace=trace.to_str(notation="infix", spot=True),
            timeout=str(timeout),
        )
        pb_solution = self.stub.MCTrace(pb_problem)
        return TraceMCStatus(pb_solution.status.lower())

    def aag2mealy(self, circuit: str) -> str:
        pb_problem = aiger_pb2.AigerCircuit(circuit=circuit)
        pb_solution = self.stub.aag2mealy(pb_problem)
        return str(pb_solution.machine)

    def mealy2aag(self, mealy: str) -> str:
        pb_problem = mealy_pb2.MealyMachine(machine=mealy)
        pb_solution = self.stub.mealy2aag(pb_problem)
        return str(pb_solution.circuit)

    def extractTransitions(self, mealy: str) -> List:
        pb_problem = mealy_pb2.MealyMachine(machine=mealy)
        pb_solution = self.stub.extractTransitions(pb_problem)
        return json.loads(pb_solution.transitions)

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
