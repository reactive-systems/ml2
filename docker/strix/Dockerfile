FROM --platform=linux/amd64 ubuntu:jammy

RUN apt-get -q update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -q install -y \
    build-essential \
    curl \
    git \
    clang-14 \
    zlib1g-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ARG GRAALVM_VERSION=21.0.0.2

ENV GRAALVM_PKG=https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-$GRAALVM_VERSION/graalvm-ce-java11-linux-amd64-$GRAALVM_VERSION.tar.gz

RUN mkdir /usr/lib/jvm && \
    curl --fail --silent --location --retry 3 ${GRAALVM_PKG} \
    | gunzip \
    | tar x -C /usr/lib/jvm && \
    mv /usr/lib/jvm/graalvm-ce-java11-$GRAALVM_VERSION /usr/lib/jvm/java-11-graalvm

ENV PATH=/usr/lib/jvm/java-11-graalvm/bin:${PATH} \
    JAVA_HOME=/usr/lib/jvm/java-11-graalvm

RUN /usr/lib/jvm/java-11-graalvm/bin/gu install native-image

RUN git clone https://github.com/meyerphi/strix.git && \
    cd strix && \
    git submodule init && \
    git submodule update

RUN cd strix && \
    cargo build --release
