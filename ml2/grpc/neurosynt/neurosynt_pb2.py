# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/neurosynt/neurosynt.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ml2.grpc.ltl import ltl_syn_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2
from ml2.grpc.tools import tools_pb2 as ml2_dot_grpc_dot_tools_dot_tools__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"ml2/grpc/neurosynt/neurosynt.proto\x1a\x1aml2/grpc/ltl/ltl_syn.proto\x1a\x1aml2/grpc/tools/tools.proto2\xb4\x02\n\tNeuroSynt\x12(\n\x05Setup\x12\r.SetupRequest\x1a\x0e.SetupResponse\"\x00\x12=\n\x08Identify\x12\x16.IdentificationRequest\x1a\x17.IdentificationResponse\"\x00\x12\x35\n\nSynthesize\x12\x0e.LTLSynProblem\x1a\x15.NeuralLTLSynSolution\"\x00\x12?\n\x10SynthesizeStream\x12\x0e.LTLSynProblem\x1a\x15.NeuralLTLSynSolution\"\x00(\x01\x30\x01\x12\x46\n\x0fSynthesizeBatch\x12\x0e.LTLSynProblem\x1a\x1d.NeuralLTLSynSolutionSpecPair\"\x00(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.neurosynt.neurosynt_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_NEUROSYNT']._serialized_start=95
  _globals['_NEUROSYNT']._serialized_end=403
# @@protoc_insertion_point(module_scope)
