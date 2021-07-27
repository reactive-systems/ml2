FROM ubuntu:20.04
ARG PYTHON_VERSION=3.8.8
ARG SPOT_VERSION=2.9.6

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    wget \
    zlib1g-dev 

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure && \
    make && make install && \
    cd / && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

RUN wget http://www.lrde.epita.fr/dload/spot/spot-$SPOT_VERSION.tar.gz && \
    tar xzf spot-$SPOT_VERSION.tar.gz && \
    cd spot-$SPOT_VERSION && \
    ./configure && \
    make && \
    make install && \
    cd / && \
    rm spot-$SPOT_VERSION.tar.gz && \
    rm -r spot-$SPOT_VERSION

RUN pip3 install --upgrade pip && \
    pip3 install ipython

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"