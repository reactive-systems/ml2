# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/tools/tools.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aml2/grpc/tools/tools.proto\"t\n\x0cSetupRequest\x12\x31\n\nparameters\x18\x01 \x03(\x0b\x32\x1d.SetupRequest.ParametersEntry\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"/\n\rSetupResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"\x17\n\x15IdentificationRequest\"`\n\x16IdentificationResponse\x12\x0c\n\x04tool\x18\x01 \x01(\t\x12\'\n\x0f\x66unctionalities\x18\x02 \x03(\x0e\x32\x0e.Functionality\x12\x0f\n\x07version\x18\x03 \x01(\t*\xc7\x03\n\rFunctionality\x12\x17\n\x13\x46UNCTIONALITY_OTHER\x10\x00\x12)\n%FUNCTIONALITY_LTL_AIGER_MODELCHECKING\x10\x01\x12)\n%FUNCTIONALITY_LTL_MEALY_MODELCHECKING\x10\x02\x12%\n!FUNCTIONALITY_LTL_AIGER_SYNTHESIS\x10\x03\x12%\n!FUNCTIONALITY_LTL_MEALY_SYNTHESIS\x10\x04\x12!\n\x1d\x46UNCTIONALITY_LTL_EQUIVALENCE\x10\x05\x12)\n%FUNCTIONALITY_LTL_TRACE_MODELCHECKING\x10\x06\x12\x19\n\x15\x46UNCTIONALITY_RANDLTL\x10\x07\x12 \n\x1c\x46UNCTIONALITY_AIGER_TO_MEALY\x10\x08\x12 \n\x1c\x46UNCTIONALITY_MEALY_TO_AIGER\x10\t\x12\x1e\n\x1a\x46UNCTIONALITY_TLSF_TO_SPEC\x10\n\x12,\n(FUNCTIONALITY_NEURAL_LTL_AIGER_SYNTHESIS\x10\x0b\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.tools.tools_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SETUPREQUEST_PARAMETERSENTRY']._options = None
  _globals['_SETUPREQUEST_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_FUNCTIONALITY']._serialized_start=321
  _globals['_FUNCTIONALITY']._serialized_end=776
  _globals['_SETUPREQUEST']._serialized_start=30
  _globals['_SETUPREQUEST']._serialized_end=146
  _globals['_SETUPREQUEST_PARAMETERSENTRY']._serialized_start=97
  _globals['_SETUPREQUEST_PARAMETERSENTRY']._serialized_end=146
  _globals['_SETUPRESPONSE']._serialized_start=148
  _globals['_SETUPRESPONSE']._serialized_end=195
  _globals['_IDENTIFICATIONREQUEST']._serialized_start=197
  _globals['_IDENTIFICATIONREQUEST']._serialized_end=220
  _globals['_IDENTIFICATIONRESPONSE']._serialized_start=222
  _globals['_IDENTIFICATIONRESPONSE']._serialized_end=318
# @@protoc_insertion_point(module_scope)
