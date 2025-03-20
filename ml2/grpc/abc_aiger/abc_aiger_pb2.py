# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/abc_aiger/abc_aiger.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ml2.grpc.tools import tools_pb2 as ml2_dot_grpc_dot_tools_dot_tools__pb2
from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
from ml2.grpc.aiger import aiger_pb2 as ml2_dot_grpc_dot_aiger_dot_aiger__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"ml2/grpc/abc_aiger/abc_aiger.proto\x1a\x1aml2/grpc/tools/tools.proto\x1a\x1egoogle/protobuf/duration.proto\x1a\x1aml2/grpc/aiger/aiger.proto\"\xb4\x01\n\x1b\x43onvertBinaryToAasciRequest\x12@\n\nparameters\x18\x01 \x03(\x0b\x32,.ConvertBinaryToAasciRequest.ParametersEntry\x12 \n\x03\x61ig\x18\x02 \x01(\x0b\x32\x13.AigerBinaryCircuit\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x9b\x01\n\x1c\x43onvertBinaryToAasciResponse\x12\x1f\n\x03\x61\x61g\x18\x01 \x01(\x0b\x32\r.AigerCircuitH\x00\x88\x01\x01\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x01\x88\x01\x01\x42\x06\n\x04_aagB\x07\n\x05_time\"\xae\x01\n\x1b\x43onvertAasciToBinaryRequest\x12@\n\nparameters\x18\x01 \x03(\x0b\x32,.ConvertAasciToBinaryRequest.ParametersEntry\x12\x1a\n\x03\x61\x61g\x18\x02 \x01(\x0b\x32\r.AigerCircuit\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xa1\x01\n\x1c\x43onvertAasciToBinaryResponse\x12%\n\x03\x61ig\x18\x01 \x01(\x0b\x32\x13.AigerBinaryCircuitH\x00\x88\x01\x01\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x01\x88\x01\x01\x42\x06\n\x04_aigB\x07\n\x05_time\"\xa8\x01\n\x18\x43onvertAigerToDotRequest\x12=\n\nparameters\x18\x01 \x03(\x0b\x32).ConvertAigerToDotRequest.ParametersEntry\x12\x1a\n\x03\x61\x61g\x18\x02 \x01(\x0b\x32\r.AigerCircuit\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x89\x01\n\x19\x43onvertAigerToDotResponse\x12\x10\n\x03\x64ot\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x01\x88\x01\x01\x42\x06\n\x04_dotB\x07\n\x05_time\"x\n\x17\x43onvertDotToSvgResponse\x12\x10\n\x03svg\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x01\x88\x01\x01\x42\x06\n\x04_svgB\x07\n\x05_time\" \n\x11\x43onvertDotRequest\x12\x0b\n\x03\x64ot\x18\x01 \x01(\t\"x\n\x17\x43onvertDotToPngResponse\x12\x10\n\x03png\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x01\x88\x01\x01\x42\x06\n\x04_pngB\x07\n\x05_time\"\xbb\x01\n\x14\x41igerSimplifyRequest\x12\x39\n\nparameters\x18\x01 \x03(\x0b\x32%.AigerSimplifyRequest.ParametersEntry\x12\x19\n\x11simplify_commands\x18\x02 \x03(\t\x12\x1a\n\x03\x61\x61g\x18\x03 \x01(\x0b\x32\r.AigerCircuit\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x87\x01\n\x15\x41igerSimplifyResponse\x12\x1a\n\x03\x61\x61g\x18\x01 \x03(\x0b\x32\r.AigerCircuit\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x00\x88\x01\x01\x42\x07\n\x05_time2\xb7\x04\n\x08\x41\x42\x43\x41iger\x12(\n\x05Setup\x12\r.SetupRequest\x1a\x0e.SetupResponse\"\x00\x12=\n\x08Identify\x12\x16.IdentificationRequest\x1a\x17.IdentificationResponse\"\x00\x12U\n\x14\x43onvertBinaryToAasci\x12\x1c.ConvertBinaryToAasciRequest\x1a\x1d.ConvertBinaryToAasciResponse\"\x00\x12U\n\x14\x43onvertAasciToBinary\x12\x1c.ConvertAasciToBinaryRequest\x1a\x1d.ConvertAasciToBinaryResponse\"\x00\x12L\n\x11\x43onvertAigerToDot\x12\x19.ConvertAigerToDotRequest\x1a\x1a.ConvertAigerToDotResponse\"\x00\x12\x41\n\x0f\x43onvertDotToSvg\x12\x12.ConvertDotRequest\x1a\x18.ConvertDotToSvgResponse\"\x00\x12\x41\n\x0f\x43onvertDotToPng\x12\x12.ConvertDotRequest\x1a\x18.ConvertDotToPngResponse\"\x00\x12@\n\rAigerSimplify\x12\x15.AigerSimplifyRequest\x1a\x16.AigerSimplifyResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.abc_aiger.abc_aiger_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_CONVERTBINARYTOAASCIREQUEST_PARAMETERSENTRY']._options = None
  _globals['_CONVERTBINARYTOAASCIREQUEST_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_CONVERTAASCITOBINARYREQUEST_PARAMETERSENTRY']._options = None
  _globals['_CONVERTAASCITOBINARYREQUEST_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_CONVERTAIGERTODOTREQUEST_PARAMETERSENTRY']._options = None
  _globals['_CONVERTAIGERTODOTREQUEST_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_AIGERSIMPLIFYREQUEST_PARAMETERSENTRY']._options = None
  _globals['_AIGERSIMPLIFYREQUEST_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_CONVERTBINARYTOAASCIREQUEST']._serialized_start=127
  _globals['_CONVERTBINARYTOAASCIREQUEST']._serialized_end=307
  _globals['_CONVERTBINARYTOAASCIREQUEST_PARAMETERSENTRY']._serialized_start=258
  _globals['_CONVERTBINARYTOAASCIREQUEST_PARAMETERSENTRY']._serialized_end=307
  _globals['_CONVERTBINARYTOAASCIRESPONSE']._serialized_start=310
  _globals['_CONVERTBINARYTOAASCIRESPONSE']._serialized_end=465
  _globals['_CONVERTAASCITOBINARYREQUEST']._serialized_start=468
  _globals['_CONVERTAASCITOBINARYREQUEST']._serialized_end=642
  _globals['_CONVERTAASCITOBINARYREQUEST_PARAMETERSENTRY']._serialized_start=258
  _globals['_CONVERTAASCITOBINARYREQUEST_PARAMETERSENTRY']._serialized_end=307
  _globals['_CONVERTAASCITOBINARYRESPONSE']._serialized_start=645
  _globals['_CONVERTAASCITOBINARYRESPONSE']._serialized_end=806
  _globals['_CONVERTAIGERTODOTREQUEST']._serialized_start=809
  _globals['_CONVERTAIGERTODOTREQUEST']._serialized_end=977
  _globals['_CONVERTAIGERTODOTREQUEST_PARAMETERSENTRY']._serialized_start=258
  _globals['_CONVERTAIGERTODOTREQUEST_PARAMETERSENTRY']._serialized_end=307
  _globals['_CONVERTAIGERTODOTRESPONSE']._serialized_start=980
  _globals['_CONVERTAIGERTODOTRESPONSE']._serialized_end=1117
  _globals['_CONVERTDOTTOSVGRESPONSE']._serialized_start=1119
  _globals['_CONVERTDOTTOSVGRESPONSE']._serialized_end=1239
  _globals['_CONVERTDOTREQUEST']._serialized_start=1241
  _globals['_CONVERTDOTREQUEST']._serialized_end=1273
  _globals['_CONVERTDOTTOPNGRESPONSE']._serialized_start=1275
  _globals['_CONVERTDOTTOPNGRESPONSE']._serialized_end=1395
  _globals['_AIGERSIMPLIFYREQUEST']._serialized_start=1398
  _globals['_AIGERSIMPLIFYREQUEST']._serialized_end=1585
  _globals['_AIGERSIMPLIFYREQUEST_PARAMETERSENTRY']._serialized_start=258
  _globals['_AIGERSIMPLIFYREQUEST_PARAMETERSENTRY']._serialized_end=307
  _globals['_AIGERSIMPLIFYRESPONSE']._serialized_start=1588
  _globals['_AIGERSIMPLIFYRESPONSE']._serialized_end=1723
  _globals['_ABCAIGER']._serialized_start=1726
  _globals['_ABCAIGER']._serialized_end=2293
# @@protoc_insertion_point(module_scope)
