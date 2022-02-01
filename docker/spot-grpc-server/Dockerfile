ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/spot

RUN pip3 install --upgrade pip && \
    pip3 install grpcio-tools

COPY ml2/tools/protos ml2/tools/protos
COPY ml2/ltl/ltl_sat/ltl_sat_status.py ml2/ltl/ltl_sat/
COPY ml2/trace/trace_mc_status.py ml2/trace/
COPY ml2/tools/spot/spot_grpc_server.py \
    ml2/tools/spot/spot_pb2.py \
    ml2/tools/spot/spot_pb2_grpc.py \
    ml2/tools/spot/spot_wrapper.py \
    ml2/tools/spot/

ENTRYPOINT [ "python3", "-m", "ml2.tools.spot.spot_grpc_server"]

