FROM projects.cispa.saarland:5005/group-finkbeiner/tools/ml2/strix

ENV PATH=/root/.local/bin:${PATH}

RUN apt-get -q update && \
    apt-get -q install -y \
    gettext-base \
    haskell-stack \
    sudo \
    wget && \
    stack upgrade

RUN cd /strix/scripts && \
    yes Y | ./install_dependencies.sh

RUN git clone https://github.com/reactive-systems/syfco.git && \
    cd syfco && \
    stack install