# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/ltl/ltl_sat.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aml2/grpc/ltl/ltl_sat.proto\"C\n\rLTLSatProblem\x12\x0f\n\x07\x66ormula\x18\x01 \x01(\t\x12\x10\n\x08simplify\x18\x02 \x01(\x08\x12\x0f\n\x07timeout\x18\x03 \x01(\x02\"/\n\x0eLTLSatSolution\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\r\n\x05trace\x18\x02 \x01(\tb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.ltl.ltl_sat_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_LTLSATPROBLEM']._serialized_start=30
  _globals['_LTLSATPROBLEM']._serialized_end=97
  _globals['_LTLSATSOLUTION']._serialized_start=99
  _globals['_LTLSATSOLUTION']._serialized_end=146
# @@protoc_insertion_point(module_scope)
