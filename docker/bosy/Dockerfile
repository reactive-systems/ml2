FROM swift:5.3.3-bionic

RUN export DEBIAN_FRONTEND=noninteractive && apt-get -q update && \
    apt-get -q install -y \
    bison \
    build-essential \
    clang \
    cmake \
    curl \
    flex \
    gcc \
    git \
    libantlr3c-dev \
    libbdd-dev \
    libboost-program-options-dev \
    libicu-dev \
    libreadline-dev \
    mercurial \
    psmisc \
    unzip \
    vim-common \
    wget \
    zlib1g-dev \
    libsqlite3-dev

RUN curl -sSL https://get.haskellstack.org/ | sh

ENV PATH=/root/.local/bin:${PATH}

RUN git clone https://github.com/frederikschmitt/bosy.git && \
    cd bosy && \
    make