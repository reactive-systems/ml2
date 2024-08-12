ARG CONTAINER_REGISTRY=ghcr.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/booleforce

RUN apt-get -q update && \
    apt-get -q install -y \
    python3 \
    python3-pip

COPY ml2 ml2/ml2
COPY LICENSE pyproject.toml setup.cfg ml2/
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir ml2/


ENTRYPOINT [ "python3", "-m", "ml2.tools.booleforce.booleforce_grpc_server"]