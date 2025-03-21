FROM --platform=amd64 tensorflow/tensorflow:2.16.2-gpu

# Google Cloud SDK

# ENV CLOUDSDK_PYTHON=/usr/bin/python3

RUN curl -fsSL -o google-cloud-cli.tar.gz  https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && \
    tar -xzf google-cloud-cli.tar.gz -C /usr/local && \
    rm google-cloud-cli.tar.gz && \
    /usr/local/google-cloud-sdk/install.sh -q --path-update true

# Docker engine

RUN curl -fsSL -o get-docker.sh https://get.docker.com  && \
    sh ./get-docker.sh && \
    rm get-docker.sh

# PyPI dependencies

RUN pip --no-cache-dir install --upgrade pip
# Installing backwards-compatible tf-keras package because Transformers does not yet support Keras 3
RUN pip install tf-keras==2.16
RUN pip --no-cache-dir install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip --no-cache-dir install datasets docker google-cloud-storage grpcio jupyter matplotlib nltk numpy pandas ray[default] sly transformers[torch] tqdm wandb
