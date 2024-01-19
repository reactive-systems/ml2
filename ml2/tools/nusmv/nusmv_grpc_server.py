"""gRPC Server that model checks LTL specifications and AIGER circuits using NuSMV"""

import argparse
import logging
from concurrent import futures
from datetime import datetime
from typing import Generator

import grpc

from ...grpc import tools_pb2
from ...grpc.ltl.ltl_mc_pb2 import LTLMCProblem, LTLMCSolution
from ...grpc.nusmv import nusmv_pb2_grpc
from ..ltl_tool.tool_ltl_mc_problem import ToolLTLMCProblem, ToolLTLMCSolution
from .nusmv_wrapper import nusmv_wrapper

AIG_TO_SMV_PATH = "/aiger/aigtosmv"
LTLFILT_PATH = "/spot-2.11.5/bin/ltlfilt"
NUSMV_PATH = "/NuSMV-2.6.0-Linux/bin/NuSMV"
TEMP_DIR = "/tmp"

logger = logging.getLogger("NuSMV gRPC Server")


class NuSMVService(nusmv_pb2_grpc.NuSMVServicer):
    def ModelCheck(self, request: LTLMCProblem, context) -> LTLMCSolution:
        start = datetime.now()
        timeout = float(request.parameters["timeout"]) if "timeout" in request.parameters else None
        problem = ToolLTLMCProblem.from_pb2_LTLMCProblem(request)
        assert problem.circuit is not None
        result = nusmv_wrapper(
            spec=problem.specification,
            circuit=problem.circuit,
            realizable=problem.realizable,
            aig_to_smv_path=AIG_TO_SMV_PATH,
            ltlfilt_path=LTLFILT_PATH,
            nusmv_path=NUSMV_PATH,
            temp_dir=TEMP_DIR,
            timeout=timeout,
        )
        duration = datetime.now() - start
        print(f"Model checking took {duration}")
        return ToolLTLMCSolution(
            status=result["status"],
            time=duration,
            tool="NuSMV",
            detailed_status=result["detailed_status"] if "detailed_status" in result else "",
            counterexample=result["counterexample"] if "counterexample" in result else None,
        ).to_pb2_LTLMCSolution()

    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def ModelCheckStream(self, request_iterator, context) -> Generator[LTLMCSolution, None, None]:
        for request in request_iterator:
            yield self.ModelCheck(request, context)

    def Identify(self, request, context):
        return tools_pb2.IdentificationResponse(
            tool="NuSMV",
            functionalities=[tools_pb2.FUNCTIONALITY_LTL_AIGER_MODELCHECKING],
            version="2.6.0",
        )


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    nusmv_pb2_grpc.add_NuSMVServicer_to_server(NuSMVService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NuSMV gRPC server")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=50051,
        metavar="port number",
        help=("port on which server accepts RPCs"),
    )
    args = parser.parse_args()
    serve(args.port)
