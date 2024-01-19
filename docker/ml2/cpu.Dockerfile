ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/deps:cpu

COPY . /ml2

RUN pip --no-cache-dir install /ml2