FROM ubuntu:24.04

ARG SPOT_VERSION=2.11.6

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    build-essential \
    python3-pip \
    wget

RUN wget http://www.lrde.epita.fr/dload/spot/spot-$SPOT_VERSION.tar.gz && \
    tar xzf spot-$SPOT_VERSION.tar.gz && \
    cd spot-$SPOT_VERSION && \
    ./configure --with-pythondir=/usr/local/lib/python3.12/dist-packages && \
    make && \
    make install && \
    cd / && \
    rm spot-$SPOT_VERSION.tar.gz && \
    rm -r spot-$SPOT_VERSION

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"