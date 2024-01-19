ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/spot

COPY ml2 ml2/ml2
COPY LICENSE pyproject.toml setup.cfg ml2/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ml2/

ENTRYPOINT [ "python3", "-m", "ml2.tools.spot.spot_grpc_server"]
