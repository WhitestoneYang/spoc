# Set base image
FROM continuumio/miniconda3

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# Set working directory
WORKDIR /workspace

# Install system packages and clear the apt cache in the same layer
RUN apt-get update && \
    apt-get install -y \
        git \
        curl \
        wget \
        pandoc \
        software-properties-common \
        openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install openbabel
RUN conda install -y -c conda-forge openbabel

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt && \
    pip install .
