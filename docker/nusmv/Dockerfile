FROM ubuntu:jammy

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    apt-get -q install -y \
    build-essential \
    git \
    python3-pip \
    wget

RUN git clone https://github.com/frederikschmitt/aiger.git && \
    cd aiger && \
    ./configure.sh && \
    make

RUN wget http://www.lrde.epita.fr/dload/spot/spot-2.11.5.tar.gz  && \
    tar xzf spot-2.11.5.tar.gz && \
    rm spot-2.11.5.tar.gz && \
    cd spot-2.11.5 && \
    ./configure && \
    make

RUN wget https://nusmv.fbk.eu/distrib/NuSMV-2.6.0-linux64.tar.gz && \
    tar xzf NuSMV-2.6.0-linux64.tar.gz && \
    rm NuSMV-2.6.0-linux64.tar.gz


