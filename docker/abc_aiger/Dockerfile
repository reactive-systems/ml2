FROM ubuntu:22.04

RUN apt-get -q update \
    && apt-get -q upgrade -y \
    && apt-get install git build-essential libreadline-dev graphviz -y

RUN git clone https://github.com/berkeley-abc/abc.git \
    && cd abc \
    && make -j 4

RUN git clone https://github.com/arminbiere/lingeling.git && \
    cd lingeling && \
    ./configure.sh && \
    make -j 4

RUN git clone https://github.com/arminbiere/aiger.git \
    && cd aiger \
    && ./configure.sh \
    && make -j 4


