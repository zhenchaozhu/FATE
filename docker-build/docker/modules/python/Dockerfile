# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/python

ARG version

COPY  python/ ./
COPY  eggroll/ ./
COPY  fml_agent ./fml_agent
COPY  start.sh ./


RUN echo "source /data/projects/python/venv/bin/activate" >> ~/.bashrc && \
    mkdir -p ./fml_agent/data

ENV PYTHONPATH=/data/projects/fate/python
CMD sleep 5; source /data/projects/python/venv/bin/activate && bash ./start.sh
