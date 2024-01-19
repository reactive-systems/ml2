FROM ubuntu:20.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    build-essential \
    git

RUN git clone https://github.com/arminbiere/lingeling.git && \
    cd lingeling && \
    ./configure.sh && \
    make