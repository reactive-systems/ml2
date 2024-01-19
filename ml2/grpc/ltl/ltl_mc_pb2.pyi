"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.duration_pb2
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import ml2.grpc.aiger.aiger_pb2
import ml2.grpc.ltl.ltl_pb2
import ml2.grpc.mealy.mealy_pb2
import ml2.grpc.trace.trace_pb2
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _LTLMCStatus:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _LTLMCStatusEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_LTLMCStatus.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    LTLMCSTATUS_UNSPECIFIED: _LTLMCStatus.ValueType  # 0
    """Default Value, additional Information should be given if set on purpose"""
    LTLMCSTATUS_SATISFIED: _LTLMCStatus.ValueType  # 1
    """Signals that the system satisfies the LTL Specification"""
    LTLMCSTATUS_VIOLATED: _LTLMCStatus.ValueType  # 2
    """Signals that the system violates the LTL Specification"""
    LTLMCSTATUS_ERROR: _LTLMCStatus.ValueType  # 3
    """Signals that some error happened during computation. Additional
    information should be given.
    """
    LTLMCSTATUS_TIMEOUT: _LTLMCStatus.ValueType  # 4
    """Signals that the model checking timed out."""
    LTLMCSTATUS_INVALID: _LTLMCStatus.ValueType  # 5
    """Signals that the system was invalid."""
    LTLMCSTATUS_NONSUCCESS: _LTLMCStatus.ValueType  # 6
    """Signals that model checking was not successful, However no error, timeout,
    invalid happened. Additional Information should be given.
    """

class LTLMCStatus(_LTLMCStatus, metaclass=_LTLMCStatusEnumTypeWrapper):
    """Mirrors ml2.ltl.ltl_mc.ltl_mc_status.LTLMCStatus"""

LTLMCSTATUS_UNSPECIFIED: LTLMCStatus.ValueType  # 0
"""Default Value, additional Information should be given if set on purpose"""
LTLMCSTATUS_SATISFIED: LTLMCStatus.ValueType  # 1
"""Signals that the system satisfies the LTL Specification"""
LTLMCSTATUS_VIOLATED: LTLMCStatus.ValueType  # 2
"""Signals that the system violates the LTL Specification"""
LTLMCSTATUS_ERROR: LTLMCStatus.ValueType  # 3
"""Signals that some error happened during computation. Additional
information should be given.
"""
LTLMCSTATUS_TIMEOUT: LTLMCStatus.ValueType  # 4
"""Signals that the model checking timed out."""
LTLMCSTATUS_INVALID: LTLMCStatus.ValueType  # 5
"""Signals that the system was invalid."""
LTLMCSTATUS_NONSUCCESS: LTLMCStatus.ValueType  # 6
"""Signals that model checking was not successful, However no error, timeout,
invalid happened. Additional Information should be given.
"""
global___LTLMCStatus = LTLMCStatus

@typing_extensions.final
class LTLMCProblem(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class ParametersEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    PARAMETERS_FIELD_NUMBER: builtins.int
    DECOMP_SPECIFICATION_FIELD_NUMBER: builtins.int
    FORMULA_SPECIFICATION_FIELD_NUMBER: builtins.int
    CIRCUIT_FIELD_NUMBER: builtins.int
    MEALY_MACHINE_FIELD_NUMBER: builtins.int
    REALIZABLE_FIELD_NUMBER: builtins.int
    @property
    def parameters(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]:
        """Defines run- and tool-specific parameters. As Map (Dict in Python).
        Typical examples are threads, timeouts etc. Can be empty.
        """
    @property
    def decomp_specification(self) -> ml2.grpc.ltl.ltl_pb2.DecompLTLSpecification: ...
    @property
    def formula_specification(self) -> ml2.grpc.ltl.ltl_pb2.LTLSpecification: ...
    @property
    def circuit(self) -> ml2.grpc.aiger.aiger_pb2.AigerCircuit: ...
    @property
    def mealy_machine(self) -> ml2.grpc.mealy.mealy_pb2.MealyMachine: ...
    realizable: builtins.bool
    """Shows, whether the specification has been found to be realizable or
    unrealizable.
    """
    def __init__(
        self,
        *,
        parameters: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        decomp_specification: ml2.grpc.ltl.ltl_pb2.DecompLTLSpecification | None = ...,
        formula_specification: ml2.grpc.ltl.ltl_pb2.LTLSpecification | None = ...,
        circuit: ml2.grpc.aiger.aiger_pb2.AigerCircuit | None = ...,
        mealy_machine: ml2.grpc.mealy.mealy_pb2.MealyMachine | None = ...,
        realizable: builtins.bool = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["circuit", b"circuit", "decomp_specification", b"decomp_specification", "formula_specification", b"formula_specification", "mealy_machine", b"mealy_machine", "specification", b"specification", "system", b"system"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["circuit", b"circuit", "decomp_specification", b"decomp_specification", "formula_specification", b"formula_specification", "mealy_machine", b"mealy_machine", "parameters", b"parameters", "realizable", b"realizable", "specification", b"specification", "system", b"system"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["specification", b"specification"]) -> typing_extensions.Literal["decomp_specification", "formula_specification"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["system", b"system"]) -> typing_extensions.Literal["circuit", "mealy_machine"] | None: ...

global___LTLMCProblem = LTLMCProblem

@typing_extensions.final
class LTLMCSolution(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    DETAILED_STATUS_FIELD_NUMBER: builtins.int
    TOOL_FIELD_NUMBER: builtins.int
    COUNTEREXAMPLE_FIELD_NUMBER: builtins.int
    TIME_FIELD_NUMBER: builtins.int
    status: global___LTLMCStatus.ValueType
    """A status that includes useful information about the run. For some status
    values, additional information should be given in detailed_status.
    """
    detailed_status: builtins.str
    """Here additional information should be supplied if the status value requires
    more details. For example an error trace for ERROR or a reason for
    NONSUCCESS
    """
    tool: builtins.str
    """Tool that created the response"""
    @property
    def counterexample(self) -> ml2.grpc.trace.trace_pb2.Trace:
        """A trace, proving the violation of the specification. Should be given if
        status is VIOLATED
        """
    @property
    def time(self) -> google.protobuf.duration_pb2.Duration:
        """How long the tool took to create the result."""
    def __init__(
        self,
        *,
        status: global___LTLMCStatus.ValueType = ...,
        detailed_status: builtins.str = ...,
        tool: builtins.str = ...,
        counterexample: ml2.grpc.trace.trace_pb2.Trace | None = ...,
        time: google.protobuf.duration_pb2.Duration | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_counterexample", b"_counterexample", "_time", b"_time", "counterexample", b"counterexample", "time", b"time"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_counterexample", b"_counterexample", "_time", b"_time", "counterexample", b"counterexample", "detailed_status", b"detailed_status", "status", b"status", "time", b"time", "tool", b"tool"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_counterexample", b"_counterexample"]) -> typing_extensions.Literal["counterexample"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_time", b"_time"]) -> typing_extensions.Literal["time"] | None: ...

global___LTLMCSolution = LTLMCSolution
