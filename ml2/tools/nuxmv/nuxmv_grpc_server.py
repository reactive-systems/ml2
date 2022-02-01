"""gRPC Server that model checks LTL specifications and AIGER circuits using nuXmv"""
import argparse
from concurrent import futures
import logging
import time

import grpc

from . import nuxmv_pb2
from . import nuxmv_pb2_grpc
from .nuxmv_wrapper import nuxmv_wrapper_dict

STRIX_PATH = "/strix"
TEMP_DIR = "/tmp"

logger = logging.getLogger("nuXmv gRPC Server")


class nuXmvServicer(nuxmv_pb2_grpc.nuXmvServicer):
    def ModelCheck(self, request, context):
        start = time.time()
        spec_pb = request.specification
        spec_dict = {
            "inputs": spec_pb.inputs,
            "outputs": spec_pb.outputs,
            "guarantees": spec_pb.guarantees,
            "assumptions": spec_pb.assumptions,
        }
        result = nuxmv_wrapper_dict(
            spec_dict,
            request.system,
            request.realizable,
            STRIX_PATH,
            TEMP_DIR,
            request.timeout if request.timeout > 0 else None,
        )
        print(f"Model checking took {time.time() - start} seconds")
        return nuxmv_pb2.Solution(status=nuxmv_pb2.Solution.Status.Value(result.value.upper()))

    def ModelCheckStream(self, request_iterator, context):
        for request in request_iterator:
            yield self.ModelCheck(request, context)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    nuxmv_pb2_grpc.add_nuXmvServicer_to_server(nuXmvServicer(), server)
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
