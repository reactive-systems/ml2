"""gRPC Server that checks the satisfiability of an LTL formula using Aalta"""

import argparse
from concurrent import futures
import logging
import time

import grpc

from . import limboole_pb2_grpc
from .limboole_wrapper import limboole_sat_wrapper, limboole_valid_wrapper
from ..protos import prop_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LimbooleServicer(limboole_pb2_grpc.LimbooleServicer):
    def CheckSat(self, request, context):
        start = time.time()
        solution = limboole_sat_wrapper(
            formula=request.formula,
            timeout=request.timeout if request.timeout > 0 else None,
        )
        end = time.time()
        print(f"Checking satisfiability took {end - start} seconds")
        return prop_pb2.PropSatSolution(
            status=solution["status"].value, assignment=solution.get("assignment", None)
        )

    def CheckValid(self, request, context):
        start = time.time()
        solution = limboole_valid_wrapper(
            formula=request.formula,
            timeout=request.timeout if request.timeout > 0 else None,
        )
        end = time.time()
        print(f"Checking validity took {end - start} seconds")
        return prop_pb2.PropSatSolution(
            status=solution["status"].value, assignment=solution.get("assignment", None)
        )


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    limboole_pb2_grpc.add_LimbooleServicer_to_server(LimbooleServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limboole gRPC server")
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
