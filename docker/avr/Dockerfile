FROM ubuntu:jammy

RUN apt-get -q update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -q install -y \
    git \
    python-is-python3 \
    sudo \
    tzdata \
    wget

RUN git clone https://github.com/aman-goel/avr.git

RUN cd avr && \
    chmod +x build.sh && \
    chmod +x deps/build_deps.sh && \
    chmod +x build/avr && \
    yes | ./build.sh