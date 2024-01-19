ARG CONTAINER_REGISTRY=ghrc.io/reactive-systems/ml2

FROM $CONTAINER_REGISTRY/deps:gpu

COPY LICENSE \
    README.rst \
    pyproject.toml \
    setup.cfg \
    ml2/

COPY ml2 ml2/ml2

RUN pip --no-cache-dir install /ml2[full]

ENV ML2_GCP_BUCKET=ml2-public

ENTRYPOINT [ "python3", "-m", "ml2.tools.neurosynt.neurosynt_grpc_server"]