"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class Trace(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRACE_FIELD_NUMBER: builtins.int
    trace: builtins.str
    def __init__(
        self,
        *,
        trace: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["trace", b"trace"]) -> None: ...

global___Trace = Trace
