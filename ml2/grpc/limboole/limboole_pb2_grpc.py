# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from ml2.grpc.prop import prop_pb2 as ml2_dot_grpc_dot_prop_dot_prop__pb2


class LimbooleStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CheckSat = channel.unary_unary(
                '/Limboole/CheckSat',
                request_serializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatSolution.FromString,
                )
        self.CheckValid = channel.unary_unary(
                '/Limboole/CheckValid',
                request_serializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatSolution.FromString,
                )


class LimbooleServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CheckSat(self, request, context):
        """check satisfiability
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckValid(self, request, context):
        """check validity
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LimbooleServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CheckSat': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckSat,
                    request_deserializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatSolution.SerializeToString,
            ),
            'CheckValid': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckValid,
                    request_deserializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatSolution.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Limboole', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Limboole(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CheckSat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Limboole/CheckSat',
            ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatProblem.SerializeToString,
            ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckValid(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Limboole/CheckValid',
            ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatProblem.SerializeToString,
            ml2_dot_grpc_dot_prop_dot_prop__pb2.PropSatSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
