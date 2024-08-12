FROM ubuntu:20.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    build-essential \
    wget

RUN wget http://fmv.jku.at/booleforce/booleforce-1.3.tar.gz && \
    tar xzf booleforce-1.3.tar.gz && \
    rm booleforce-1.3.tar.gz && \
    cd booleforce-1.3 && \
    ./configure --trace && \
    make