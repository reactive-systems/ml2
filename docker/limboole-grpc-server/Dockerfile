ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/limboole

ARG PYTHON_VERSION=3.8.12

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libssl-dev \
    openssl \
    zlib1g-dev

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

COPY ml2/prop/prop_sat_status.py ml2/prop/
COPY ml2/tools/limboole/limboole_grpc_server.py \
    ml2/tools/limboole/limboole_pb2.py \
    ml2/tools/limboole/limboole_pb2_grpc.py \
    ml2/tools/limboole/limboole_wrapper.py \
    ml2/tools/limboole/
COPY ml2/tools/protos ml2/tools/protos

ENTRYPOINT [ "python3", "-m", "ml2.tools.limboole.limboole_grpc_server"]