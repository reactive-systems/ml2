ARG CONTAINER_REGISTRY=ghcr.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/abc_aiger

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libssl-dev \
    openssl \
    wget \
    python3.11 

COPY ml2 ml2/ml2
COPY LICENSE pyproject.toml setup.cfg ml2/
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install pydot && \
    # python3.11 -m pip install --no-cache-dir ml2/
    python3.11 -m pip install -e ml2/


ENTRYPOINT [ "python3.11", "-m", "ml2.tools.abc_aiger.abc_aiger_grpc_server"]

