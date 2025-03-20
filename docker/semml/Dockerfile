FROM ubuntu:22.04

ENV LANG=C.UTF-8

ARG TARGETPLATFORM

RUN apt-get -q update \
    && apt-get -q upgrade -y \
    && apt-get install openjdk-17-jre unzip git -y

RUN git clone https://gitlab.com/live-lab/software/semml.git \
    && cd semml \
    && git checkout artifact_tacas

# Needs to be in one RUN command to avoid dynamic setting of JAVA_PLATFORM as ARG
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    JAVA_PLATFORM="aarch64"; \
    else \
    JAVA_PLATFORM="amd64"; \
    fi && \
    PATH=/usr/lib/jvm/java-17-openjdk-${JAVA_PLATFORM}/bin:${PATH} && \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-${JAVA_PLATFORM} && \
    cd semml && \
    ./gradlew distZip && \
    unzip build/distributions/semml-dev.zip

# semml at /semml/semml-dev/bin/semml semmlMain --env i0,i1,i2 --sys o0,o1,o2 --formula 'G (i0 <-> o0)'