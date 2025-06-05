FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install python and required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command to run when the container starts
CMD ["python3", "-m", "solar_knowledge.eval_full", "--help"]
