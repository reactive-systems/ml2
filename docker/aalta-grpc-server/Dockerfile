ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/aalta

ARG PYTHON_VERSION=3.8.8

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libssl-dev \
    openssl \
    wget

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make && make install && \
    cd / && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

RUN pip3 install --upgrade pip && \
    pip3 install grpcio-tools

COPY ml2/ltl/ltl_sat/ltl_sat_status.py ml2/ltl/ltl_sat/
COPY ml2/tools/aalta/aalta_grpc_server.py \
    ml2/tools/aalta/aalta_pb2.py \
    ml2/tools/aalta/aalta_pb2_grpc.py \
    ml2/tools/aalta/aalta_wrapper.py \
    ml2/tools/aalta/
COPY ml2/tools/protos ml2/tools/protos

ENTRYPOINT [ "python3", "-m", "ml2.tools.aalta.aalta_grpc_server"]