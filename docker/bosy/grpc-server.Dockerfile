ARG CONTAINER_REGISTRY=ghcr.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/bosy

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libssl-dev \
    openssl \
    wget \
    python3 \
    python3-pip

# copying ml2 files into BoSy directory as BoSy can only be started in its own directory

WORKDIR /root/bosy

COPY ml2 ml2/ml2
COPY LICENSE pyproject.toml setup.cfg ml2/
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir ml2/

ENTRYPOINT [ "python3", "-m", "ml2.tools.bosy.bosy_grpc_server"]

