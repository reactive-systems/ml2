"""gRPC Server that model checks LTL specifications and AIGER circuits using nuXmv"""

import argparse
import logging
from concurrent import futures
from datetime import datetime
from typing import Generator

import grpc

from ...grpc import tools_pb2
from ...grpc.ltl.ltl_mc_pb2 import LTLMCProblem, LTLMCSolution
from ...grpc.nuxmv import nuxmv_pb2_grpc
from ..ltl_tool.tool_ltl_mc_problem import ToolLTLMCProblem, ToolLTLMCSolution
from .nuxmv_wrapper import nuxmv_wrapper

STRIX_PATH = "/strix"
TEMP_DIR = "/tmp"

logger = logging.getLogger("nuXmv gRPC Server")


class NuxmvService(nuxmv_pb2_grpc.NuxmvServicer):
    def ModelCheck(self, request: LTLMCProblem, context) -> LTLMCSolution:
        start = datetime.now()
        timeout = float(request.parameters["timeout"]) if "timeout" in request.parameters else None
        problem = ToolLTLMCProblem.from_pb2_LTLMCProblem(request)
        assert problem.circuit is not None
        status, detailed_status = nuxmv_wrapper(
            spec=problem.specification,
            circuit=problem.circuit,
            realizable=problem.realizable,
            strix_path=STRIX_PATH,
            temp_dir=TEMP_DIR,
            timeout=timeout,
        )
        duration = datetime.now() - start
        print(f"Model checking took {duration}")
        return ToolLTLMCSolution(
            status=status,
            time=duration,
            tool="nuXmv",
            detailed_status=detailed_status,
        ).to_pb2_LTLMCSolution()

    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def ModelCheckStream(self, request_iterator, context) -> Generator[LTLMCSolution, None, None]:
        for request in request_iterator:
            yield self.ModelCheck(request, context)

    def Identify(self, request, context):
        return tools_pb2.IdentificationResponse(
            tool="nuXmv",
            functionalities=[tools_pb2.FUNCTIONALITY_LTL_AIGER_MODELCHECKING],
            version="2.6.0",
        )


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    nuxmv_pb2_grpc.add_NuxmvServicer_to_server(NuxmvService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nuXmv gRPC server")
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
