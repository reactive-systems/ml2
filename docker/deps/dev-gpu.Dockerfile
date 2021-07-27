FROM projects.cispa.saarland:5005/group-finkbeiner/tools/ml2/deps:latest-gpu

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    git-all

RUN pip --no-cache-dir install grpcio-tools pylint rstcheck sphinx yapf
