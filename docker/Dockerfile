################################################################################################
# Base stage
# - CUDA 12.8 Runtime + Python 3.10 on Ubuntu 22.04
################################################################################################
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

LABEL org.opencontainers.image.title="Pomelo-Tea Discord Bot (CUDA Ready)"
LABEL org.opencontainers.image.authors="yoseph.huang@gmail.com"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]


################################################################################################
# user-setup stage
# - Creates a non-root user for improved security.
################################################################################################
FROM base AS user-setup

ARG USER=dcuser
ARG USER_UID=1000

RUN useradd -m -u ${USER_UID} ${USER}


################################################################################################
# dc-bot stage
# - Set up application environment
################################################################################################
FROM user-setup AS dc-bot

WORKDIR /discord-bot

# py env & required packages
RUN apt-get update && \
    apt-get install -y \
        python3.10 \
        python3-pip \
        python3-venv \
        curl \
        ffmpeg \
        build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# pip pkgs
RUN pip install \
    discord.py[voice] \
    gTTS


################################################################################################
# llm stage
# - 若需 LLM 模型，可在此階段下載或部署
################################################################################################
FROM dc-bot AS llm

# install ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh 

# pip pkgs
RUN pip install \
    pandas \
    openpyxl \
    pyyaml \
    faiss-gpu \
    sentence-transformers \
    langchain-huggingface \
    langchain-ollama

# USER dcuser