FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends locales gcc g++ build-essential libtool autoconf unzip git wget libssl-dev ninja-build libpng-dev zlib1g-dev && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && \
    wget https://cmake.org/files/v3.30/cmake-3.30.2.tar.gz && \
    tar zxvf cmake-3.30.2.tar.gz && cd cmake-3.30.2 && ./bootstrap && make -j8 && make install && cd .. && rm cmake-3.30.2.tar.gz && rm -rf cmake-3.30.2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
