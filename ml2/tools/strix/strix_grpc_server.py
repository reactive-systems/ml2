"""gRPC Server that synthesizes LTL specifications using Strix"""

import argparse
from concurrent import futures
import logging
import time

import grpc

from . import strix_pb2
from . import strix_pb2_grpc
from .strix_wrapper import strix_wrapper_dict

logger = logging.getLogger("Strix gRPC Server")


class StrixServicer(strix_pb2_grpc.StrixServicer):
    def Synthesize(self, request, context):
        start = time.time()
        spec_pb = request.specification
        spec_dict = {
            "inputs": spec_pb.inputs,
            "outputs": spec_pb.outputs,
            "guarantees": spec_pb.guarantees,
            "assumptions": spec_pb.assumptions,
        }
        result = strix_wrapper_dict(
            spec_dict,
            request.minimize_aiger,
            request.minimize_mealy,
            request.threads if request.threads > 0 else None,
            request.timeout if request.timeout > 0 else None,
        )
        end = time.time()
        print(f"Synthesizing took {end - start} seconds")
        return strix_pb2.StrixSolution(status=result["status"].value, system=result["circuit"])

    def SynthesizeStream(self, request_iterator, context):
        for request in request_iterator:
            yield self.Synthesize(request, context)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    strix_pb2_grpc.add_StrixServicer_to_server(StrixServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strix gRPC server")
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
