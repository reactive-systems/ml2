ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/strix-opt

ARG PYTHON_VERSION=3.8.8

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    openssl

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

COPY ml2/logics/ltl/ltl_spec.py ml2/logics/ltl/
COPY ml2/ltl_mc/ltl_mc_status.py ml2/ltl_mc/
COPY ml2/tools/protos ml2/tools/protos
COPY ml2/tools/nuxmv/nuxmv_grpc_server.py \
    ml2/tools/nuxmv/nuxmv_pb2.py \
    ml2/tools/nuxmv/nuxmv_pb2_grpc.py \
    ml2/tools/nuxmv/nuxmv_wrapper.py \
    ml2/tools/nuxmv/

ENTRYPOINT [ "python3", "-m", "ml2.tools.nuxmv.nuxmv_grpc_server"]

