FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
    && ln -s /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/java-17-openjdk \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==0.11.8

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV UV_LINK_MODE=copy
