ARG CONTAINER_REGISTRY=ghcr.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/lingeling:latest

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    wget

RUN wget http://fmv.jku.at/limboole/limboole1.2.tgz && \
    tar xzf limboole1.2.tgz && \
    rm limboole1.2.tgz && \
    cd limboole1.2 && \
    ./configure.sh --lingeling && \
    make