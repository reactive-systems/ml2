FROM projects.cispa.saarland:5005/group-finkbeiner/tools/ml2/strix
ARG PYTHON_VERSION=3.8.8

RUN apt-get -q update && \
    apt-get -q install -y \
    libssl-dev \
    openssl \
    wget

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure && \
    make && make install && \
    cd / && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

RUN pip3 install --upgrade pip && \
    pip3 install grpcio-tools

COPY ml2/logics/ltl/ltl_spec.py ml2/logics/ltl/
COPY ml2/tools/protos ml2/tools/protos
COPY ml2/ltl_syn/ltl_syn_status.py ml2/ltl_syn/
COPY ml2/tools/strix/strix_grpc_server.py \
    ml2/tools/strix/strix_pb2.py \
    ml2/tools/strix/strix_pb2_grpc.py \
    ml2/tools/strix/strix_wrapper.py \
    ml2/tools/strix/

ENTRYPOINT [ "python3", "-m", "ml2.tools.strix.strix_grpc_server"]

