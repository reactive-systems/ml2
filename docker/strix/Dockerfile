FROM ubuntu:focal

RUN apt-get -q update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -q install -y \
    g++ \
    clang \
    cmake \
    curl \
    git \
    libboost-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    zlib1g-dev

ARG GRAALVM_VERSION=20.1.0

ENV GRAALVM_PKG=https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-$GRAALVM_VERSION/graalvm-ce-java11-linux-amd64-$GRAALVM_VERSION.tar.gz

RUN mkdir /usr/lib/jvm && \
    curl --fail --silent --location --retry 3 ${GRAALVM_PKG} \
    | gunzip \
    | tar x -C /usr/lib/jvm && \
    mv /usr/lib/jvm/graalvm-ce-java11-$GRAALVM_VERSION /usr/lib/jvm/java-11-graalvm

ENV PATH=/usr/lib/jvm/java-11-graalvm/bin:${PATH} \
    JAVA_HOME=/usr/lib/jvm/java-11-graalvm

RUN /usr/lib/jvm/java-11-graalvm/bin/gu install native-image

RUN git clone https://gitlab.lrz.de/i7/strix.git && \
    cd strix && \
    git submodule init && \
    git submodule update && \
    make





