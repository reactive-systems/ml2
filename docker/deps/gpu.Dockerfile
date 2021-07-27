FROM tensorflow/tensorflow:latest-gpu

RUN apt-get -q update && \
    apt-get -q install -y \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libssl-dev \
    openssl \
    wget \
    zlib1g-dev

# Python

ARG PYTHON_VERSION=3.8.8

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make && make install && \
    cd / && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

RUN ln -sf $(which python3) /usr/local/bin/python
RUN ln -sf $(which pip3) /usr/local/bin/pip

#Google Cloud SDK

ENV CLOUDSDK_PYTHON=/usr/bin/python3

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

#Docker engine

RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh ./get-docker.sh && \
    rm get-docker.sh

#PyPI dependencies

RUN pip install --upgrade pip
RUN pip --no-cache-dir install docker google-cloud-storage grpcio jupyter matplotlib nltk numpy pandas ray[default] sly tensorflow tqdm wandb