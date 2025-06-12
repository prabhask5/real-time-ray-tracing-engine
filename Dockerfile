FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    tar \
    libsdl2-dev \
    libx11-dev \
    libgl1-mesa-dev \
    libfreetype6-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory.
WORKDIR /real-time-ray-tracing-engine
