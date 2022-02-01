ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/bosy

ARG PYTHON_VERSION=3.8.8

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
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

# copying deeplogic files into BoSy directory as BoSy can only be started in its own directory

WORKDIR /bosy

COPY ml2/tools/bosy/bosy_grpc_server.py \
    ml2/tools/bosy/bosy_input.py \
    ml2/tools/bosy/bosy_pb2.py \
    ml2/tools/bosy/bosy_pb2_grpc.py \
    ml2/tools/bosy/bosy_wrapper.py \
    ml2/tools/bosy/
COPY ml2/ltl_syn/ltl_syn_status.py ml2/ltl_syn/
COPY ml2/tools/protos ml2/tools/protos

ENTRYPOINT [ "python3", "-m", "ml2.tools.bosy.bosy_grpc_server"]

