"""gRPC Server that checks the satisfiability of an LTL formula using Aalta"""

import argparse
from concurrent import futures
import logging
import time

import grpc

from . import aalta_pb2_grpc
from .aalta_wrapper import aalta_wrapper_str
from ..protos import ltl_pb2

logger = logging.getLogger("Aalta gRPC Server")


class AaltaServicer(aalta_pb2_grpc.AaltaServicer):
    def CheckSat(self, request, context):
        start = time.time()
        solution = aalta_wrapper_str(
            formula=request.formula,
            evidence=True,
            timeout=request.timeout if request.timeout > 0 else None,
        )
        end = time.time()
        print(f"Checking Satisfiability took {end - start} seconds")
        return ltl_pb2.LTLSatSolution(
            status=solution["status"].value, trace=solution.get("trace", None)
        )


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    aalta_pb2_grpc.add_AaltaServicer_to_server(AaltaServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aalta gRPC server")
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
