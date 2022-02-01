FROM ubuntu:18.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    build-essential \
    git \
    zlib1g-dev

RUN git clone https://bitbucket.org/jl86/aalta.git && \
    cd aalta && \
    make release