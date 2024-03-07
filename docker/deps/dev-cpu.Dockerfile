ARG CONTAINER_REGISTRY=ghcr.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/deps:cpu

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    git \
    screen \
    tmux

RUN pip --no-cache-dir install black flake8 flake8-quotes grpcio-tools isort mypy mypy-protobuf pre-commit pytest rbql rstcheck sphinx==4.0.2
