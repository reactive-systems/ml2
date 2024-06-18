"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import ml2.grpc.trace.trace_pb2
import sys
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class LTLEquivProblem(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FORMULA1_FIELD_NUMBER: builtins.int
    FORMULA2_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    formula1: builtins.str
    formula2: builtins.str
    timeout: builtins.float
    def __init__(
        self,
        *,
        formula1: builtins.str = ...,
        formula2: builtins.str = ...,
        timeout: builtins.float | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_timeout", b"_timeout", "timeout", b"timeout"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_timeout", b"_timeout", "formula1", b"formula1", "formula2", b"formula2", "timeout", b"timeout"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_timeout", b"_timeout"]) -> typing_extensions.Literal["timeout"] | None: ...

global___LTLEquivProblem = LTLEquivProblem

class LTLEquivSolution(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    TIME_FIELD_NUMBER: builtins.int
    EXCLUSIVE_WORD_FIELD_NUMBER: builtins.int
    status: builtins.str
    time: builtins.float
    @property
    def exclusive_word(self) -> ml2.grpc.trace.trace_pb2.Trace: ...
    def __init__(
        self,
        *,
        status: builtins.str = ...,
        time: builtins.float | None = ...,
        exclusive_word: ml2.grpc.trace.trace_pb2.Trace | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_exclusive_word", b"_exclusive_word", "_time", b"_time", "exclusive_word", b"exclusive_word", "time", b"time"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_exclusive_word", b"_exclusive_word", "_time", b"_time", "exclusive_word", b"exclusive_word", "status", b"status", "time", b"time"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_exclusive_word", b"_exclusive_word"]) -> typing_extensions.Literal["exclusive_word"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_time", b"_time"]) -> typing_extensions.Literal["time"] | None: ...

global___LTLEquivSolution = LTLEquivSolution
