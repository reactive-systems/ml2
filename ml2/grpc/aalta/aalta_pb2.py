# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/aalta/aalta.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aml2/grpc/aalta/aalta.proto\"H\n\x12LTLSatProblemAalta\x12\x0f\n\x07\x66ormula\x18\x01 \x01(\t\x12\x10\n\x08simplify\x18\x02 \x01(\x08\x12\x0f\n\x07timeout\x18\x03 \x01(\x02\"4\n\x13LTLSatSolutionAalta\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\r\n\x05trace\x18\x02 \x01(\t2@\n\x05\x41\x61lta\x12\x37\n\x08\x43heckSat\x12\x13.LTLSatProblemAalta\x1a\x14.LTLSatSolutionAalta\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.aalta.aalta_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_LTLSATPROBLEMAALTA']._serialized_start=30
  _globals['_LTLSATPROBLEMAALTA']._serialized_end=102
  _globals['_LTLSATSOLUTIONAALTA']._serialized_start=104
  _globals['_LTLSATSOLUTIONAALTA']._serialized_end=156
  _globals['_AALTA']._serialized_start=158
  _globals['_AALTA']._serialized_end=222
# @@protoc_insertion_point(module_scope)
