FROM projects.cispa.saarland:5005/group-finkbeiner/tools/ml2/deps:latest-dev-gpu

COPY . /ml2

RUN pip --no-cache-dir install -e /ml2[dev]