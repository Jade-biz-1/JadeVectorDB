# Use Ubuntu 20.04 as base image for compatibility with requirements
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libeigen3-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install latest CMake (Ubuntu 20.04 ships with an older version)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get install -y cmake

# Set up working directory
WORKDIR /app

# Copy backend source code
COPY backend/ ./backend/

# Build the application
WORKDIR /app/backend
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Expose the service port
EXPOSE 8080

# Default command
CMD ["./build/jadevectordb"]