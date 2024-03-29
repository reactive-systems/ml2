ARG CONTAINER_REGISTRY=ghcr.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/limboole

ARG PYTHON_VERSION=3.8.12

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libssl-dev \
    libffi-dev \
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

COPY ml2 ml2/ml2
COPY LICENSE pyproject.toml setup.cfg ml2/
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir ml2/

ENTRYPOINT [ "python3", "-m", "ml2.tools.limboole.limboole_grpc_server"]