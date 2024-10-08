# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from ml2.grpc.aiger import aiger_pb2 as ml2_dot_grpc_dot_aiger_dot_aiger__pb2
from ml2.grpc.ltl import ltl_equiv_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2
from ml2.grpc.ltl import ltl_mc_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2
from ml2.grpc.ltl import ltl_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__pb2
from ml2.grpc.ltl import ltl_sat_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2
from ml2.grpc.ltl import ltl_syn_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2
from ml2.grpc.ltl import ltl_trace_mc_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2
from ml2.grpc.mealy import mealy_pb2 as ml2_dot_grpc_dot_mealy_dot_mealy__pb2
from ml2.grpc.spot import spot_pb2 as ml2_dot_grpc_dot_spot_dot_spot__pb2
from ml2.grpc.tools import tools_pb2 as ml2_dot_grpc_dot_tools_dot_tools__pb2


class SpotStub(object):
    """Spot: a platform for LTL and ω-automata manipulation
    https://spot.lre.epita.fr/
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Setup = channel.unary_unary(
                '/Spot/Setup',
                request_serializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.SetupRequest.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.SetupResponse.FromString,
                )
        self.Identify = channel.unary_unary(
                '/Spot/Identify',
                request_serializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.IdentificationRequest.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.IdentificationResponse.FromString,
                )
        self.ModelCheck = channel.unary_unary(
                '/Spot/ModelCheck',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCSolution.FromString,
                )
        self.ModelCheckStream = channel.stream_stream(
                '/Spot/ModelCheckStream',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCSolution.FromString,
                )
        self.Synthesize = channel.unary_unary(
                '/Spot/Synthesize',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2.LTLSynProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2.LTLSynSolution.FromString,
                )
        self.CheckEquiv = channel.unary_unary(
                '/Spot/CheckEquiv',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.FromString,
                )
        self.Inclusion = channel.stream_stream(
                '/Spot/Inclusion',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.FromString,
                )
        self.CheckEquivRenaming = channel.unary_unary(
                '/Spot/CheckEquivRenaming',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.FromString,
                )
        self.CheckSat = channel.unary_unary(
                '/Spot/CheckSat',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2.LTLSatProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2.LTLSatSolution.FromString,
                )
        self.MCTrace = channel.unary_unary(
                '/Spot/MCTrace',
                request_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2.LTLTraceMCProblem.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2.LTLTraceMCSolution.FromString,
                )
        self.RandLTL = channel.unary_stream(
                '/Spot/RandLTL',
                request_serializer=ml2_dot_grpc_dot_spot_dot_spot__pb2.RandLTLArgs.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__pb2.LTLFormula.FromString,
                )
        self.aag2mealy = channel.unary_unary(
                '/Spot/aag2mealy',
                request_serializer=ml2_dot_grpc_dot_aiger_dot_aiger__pb2.AigerCircuit.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.FromString,
                )
        self.mealy2aag = channel.unary_unary(
                '/Spot/mealy2aag',
                request_serializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_aiger_dot_aiger__pb2.AigerCircuit.FromString,
                )
        self.extractTransitions = channel.unary_unary(
                '/Spot/extractTransitions',
                request_serializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.SerializeToString,
                response_deserializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyTransitions.FromString,
                )


class SpotServicer(object):
    """Spot: a platform for LTL and ω-automata manipulation
    https://spot.lre.epita.fr/
    """

    def Setup(self, request, context):
        """Setup call, which is typically called before the first model checking call
        has happened.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Identify(self, request, context):
        """Call to find out the identity and functionality of the server, i.e. the
        tool that is running the server and what it is supposed to do.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModelCheck(self, request, context):
        """Call to model-check a single problem
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModelCheckStream(self, request_iterator, context):
        """Call to model-check a stream of problems. Same order  of problems
        and solutions is assumed
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Synthesize(self, request, context):
        """Call to synthesize a single LTL specification
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckEquiv(self, request, context):
        """Checks whether two formulas are equivalent. If not, returns a word accepted
        by exactly one of the two formulas.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Inclusion(self, request_iterator, context):
        """Checks whether one formula is included in the other. Takes pairs of
        formulas and checks both ways. Uses efficient caching for a stream of
        pairs.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckEquivRenaming(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckSat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MCTrace(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RandLTL(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def aag2mealy(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def mealy2aag(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def extractTransitions(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SpotServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Setup': grpc.unary_unary_rpc_method_handler(
                    servicer.Setup,
                    request_deserializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.SetupRequest.FromString,
                    response_serializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.SetupResponse.SerializeToString,
            ),
            'Identify': grpc.unary_unary_rpc_method_handler(
                    servicer.Identify,
                    request_deserializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.IdentificationRequest.FromString,
                    response_serializer=ml2_dot_grpc_dot_tools_dot_tools__pb2.IdentificationResponse.SerializeToString,
            ),
            'ModelCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.ModelCheck,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCSolution.SerializeToString,
            ),
            'ModelCheckStream': grpc.stream_stream_rpc_method_handler(
                    servicer.ModelCheckStream,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCSolution.SerializeToString,
            ),
            'Synthesize': grpc.unary_unary_rpc_method_handler(
                    servicer.Synthesize,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2.LTLSynProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2.LTLSynSolution.SerializeToString,
            ),
            'CheckEquiv': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckEquiv,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.SerializeToString,
            ),
            'Inclusion': grpc.stream_stream_rpc_method_handler(
                    servicer.Inclusion,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.SerializeToString,
            ),
            'CheckEquivRenaming': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckEquivRenaming,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.SerializeToString,
            ),
            'CheckSat': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckSat,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2.LTLSatProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2.LTLSatSolution.SerializeToString,
            ),
            'MCTrace': grpc.unary_unary_rpc_method_handler(
                    servicer.MCTrace,
                    request_deserializer=ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2.LTLTraceMCProblem.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2.LTLTraceMCSolution.SerializeToString,
            ),
            'RandLTL': grpc.unary_stream_rpc_method_handler(
                    servicer.RandLTL,
                    request_deserializer=ml2_dot_grpc_dot_spot_dot_spot__pb2.RandLTLArgs.FromString,
                    response_serializer=ml2_dot_grpc_dot_ltl_dot_ltl__pb2.LTLFormula.SerializeToString,
            ),
            'aag2mealy': grpc.unary_unary_rpc_method_handler(
                    servicer.aag2mealy,
                    request_deserializer=ml2_dot_grpc_dot_aiger_dot_aiger__pb2.AigerCircuit.FromString,
                    response_serializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.SerializeToString,
            ),
            'mealy2aag': grpc.unary_unary_rpc_method_handler(
                    servicer.mealy2aag,
                    request_deserializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.FromString,
                    response_serializer=ml2_dot_grpc_dot_aiger_dot_aiger__pb2.AigerCircuit.SerializeToString,
            ),
            'extractTransitions': grpc.unary_unary_rpc_method_handler(
                    servicer.extractTransitions,
                    request_deserializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.FromString,
                    response_serializer=ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyTransitions.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Spot', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Spot(object):
    """Spot: a platform for LTL and ω-automata manipulation
    https://spot.lre.epita.fr/
    """

    @staticmethod
    def Setup(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/Setup',
            ml2_dot_grpc_dot_tools_dot_tools__pb2.SetupRequest.SerializeToString,
            ml2_dot_grpc_dot_tools_dot_tools__pb2.SetupResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Identify(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/Identify',
            ml2_dot_grpc_dot_tools_dot_tools__pb2.IdentificationRequest.SerializeToString,
            ml2_dot_grpc_dot_tools_dot_tools__pb2.IdentificationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ModelCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/ModelCheck',
            ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ModelCheckStream(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/Spot/ModelCheckStream',
            ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2.LTLMCSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Synthesize(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/Synthesize',
            ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2.LTLSynProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2.LTLSynSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckEquiv(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/CheckEquiv',
            ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Inclusion(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/Spot/Inclusion',
            ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckEquivRenaming(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/CheckEquivRenaming',
            ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2.LTLEquivSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

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
        return grpc.experimental.unary_unary(request, target, '/Spot/CheckSat',
            ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2.LTLSatProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2.LTLSatSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MCTrace(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/MCTrace',
            ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2.LTLTraceMCProblem.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2.LTLTraceMCSolution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RandLTL(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/Spot/RandLTL',
            ml2_dot_grpc_dot_spot_dot_spot__pb2.RandLTLArgs.SerializeToString,
            ml2_dot_grpc_dot_ltl_dot_ltl__pb2.LTLFormula.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def aag2mealy(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/aag2mealy',
            ml2_dot_grpc_dot_aiger_dot_aiger__pb2.AigerCircuit.SerializeToString,
            ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def mealy2aag(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/mealy2aag',
            ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.SerializeToString,
            ml2_dot_grpc_dot_aiger_dot_aiger__pb2.AigerCircuit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def extractTransitions(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Spot/extractTransitions',
            ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyMachine.SerializeToString,
            ml2_dot_grpc_dot_mealy_dot_mealy__pb2.MealyTransitions.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
