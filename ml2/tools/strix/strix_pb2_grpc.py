# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from ml2.tools.strix import strix_pb2 as ml2_dot_tools_dot_strix_dot_strix__pb2


class StrixStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Synthesize = channel.unary_unary(
            "/Strix/Synthesize",
            request_serializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixProblem.SerializeToString,
            response_deserializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixSolution.FromString,
        )
        self.SynthesizeStream = channel.stream_stream(
            "/Strix/SynthesizeStream",
            request_serializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixProblem.SerializeToString,
            response_deserializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixSolution.FromString,
        )


class StrixServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Synthesize(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SynthesizeStream(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_StrixServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Synthesize": grpc.unary_unary_rpc_method_handler(
            servicer.Synthesize,
            request_deserializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixProblem.FromString,
            response_serializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixSolution.SerializeToString,
        ),
        "SynthesizeStream": grpc.stream_stream_rpc_method_handler(
            servicer.SynthesizeStream,
            request_deserializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixProblem.FromString,
            response_serializer=ml2_dot_tools_dot_strix_dot_strix__pb2.StrixSolution.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("Strix", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Strix(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Synthesize(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/Strix/Synthesize",
            ml2_dot_tools_dot_strix_dot_strix__pb2.StrixProblem.SerializeToString,
            ml2_dot_tools_dot_strix_dot_strix__pb2.StrixSolution.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def SynthesizeStream(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/Strix/SynthesizeStream",
            ml2_dot_tools_dot_strix_dot_strix__pb2.StrixProblem.SerializeToString,
            ml2_dot_tools_dot_strix_dot_strix__pb2.StrixSolution.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
