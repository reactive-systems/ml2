"""gRPC Server that synthesizes LTL specifications using BoSy"""

import argparse
from concurrent import futures
import logging
import time

import grpc

from . import bosy_pb2
from . import bosy_pb2_grpc
from . import bosy_wrapper

BOSY_PATH = "/bosy/bosy.sh"
TEMP_DIR = "/tmp"

logger = logging.getLogger("BoSy gRPC Server")


class BoSyServicer(bosy_pb2_grpc.BoSyServicer):
    def Synthesize(self, request, context):
        start = time.time()
        spec_pb = request.specification
        spec_dict = {
            "inputs": spec_pb.inputs,
            "outputs": spec_pb.outputs,
            "guarantees": spec_pb.guarantees,
            "assumptions": spec_pb.assumptions,
        }
        result = bosy_wrapper.bosy_wrapper_dict(
            spec_dict,
            BOSY_PATH,
            request.timeout if request.timeout > 0 else None,
            request.optimize,
            TEMP_DIR,
        )
        end = time.time()
        print(f"Synthesizing took {end - start} seconds")
        return bosy_pb2.BoSySolution(status=result["status"].value, system=result["circuit"])

    def SynthesizeStream(self, request_iterator, context):
        for request in request_iterator:
            yield self.Synthesize(request, context)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    bosy_pb2_grpc.add_BoSyServicer_to_server(BoSyServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoSy gRPC server")
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
