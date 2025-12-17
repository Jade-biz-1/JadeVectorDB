###############################################################################
# JadeVectorDB Unified Dockerfile
###############################################################################
# This Dockerfile uses the unified build system with all dependencies
# fetched from source. No external package dependencies required!
#
# Build stages:
#   1. builder  - Builds the application with all dependencies from source
#   2. runtime  - Minimal runtime image with only the executable
#
# Usage:
#   docker build -f Dockerfile.unified -t jadevectordb:latest .
#   docker run -p 8080:8080 jadevectordb:latest
###############################################################################

#############################################################################
# Stage 1: Builder
#############################################################################

FROM ubuntu:24.04 AS builder

LABEL maintainer="JadeVectorDB Team"
LABEL description="High-Performance Distributed Vector Database - Builder Stage"

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and Boost development libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    libboost-all-dev \
    libssl-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN cmake --version && \
    g++ --version && \
    git --version && \
    ninja --version

# Set working directory
WORKDIR /build

# Copy backend source code
COPY backend/ /build/

# Build configuration
ENV BUILD_TYPE=Release
ENV BUILD_TESTS=OFF
ENV BUILD_BENCHMARKS=OFF
ENV BUILD_COVERAGE=OFF
ENV BUILD_WITH_GRPC=OFF
ENV CLEAN_BUILD=false
ENV PARALLEL_JOBS=4
ENV VERBOSE=false

# Run the unified build script
# All dependencies will be fetched from source via CMake FetchContent
RUN chmod +x build.sh && \
    ./build.sh --type ${BUILD_TYPE} \
               --no-tests \
               --no-benchmarks \
               --jobs ${PARALLEL_JOBS}

# Verify the build
RUN ls -lh build/jadevectordb && \
    file build/jadevectordb && \
    ldd build/jadevectordb

#############################################################################
# Stage 2: Runtime
#############################################################################
FROM ubuntu:24.04 AS runtime

LABEL maintainer="JadeVectorDB Team"
LABEL description="High-Performance Distributed Vector Database"
LABEL version="1.0.0"

# Install only runtime dependencies + curl for health checks
# Most libraries are statically linked, so we only need basic system libraries
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r jadevectordb && \
    useradd -r -g jadevectordb -s /bin/bash -d /app jadevectordb

# Set working directory
WORKDIR /app

# Copy only the executable from builder
COPY --from=builder /build/build/jadevectordb /app/jadevectordb

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R jadevectordb:jadevectordb /app

# Switch to non-root user
USER jadevectordb

# Expose default port
EXPOSE 8080

# Health check using HTTP endpoint (already implemented in T11.6.1)
# Checks every 30s, times out after 10s, starts checking after 10s, fails after 3 retries
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV JADEVECTORDB_PORT=8080 \
    JADEVECTORDB_LOG_LEVEL=info \
    JADEVECTORDB_DATA_DIR=/app/data

# Graceful shutdown timeout (Docker sends SIGTERM, waits 10s by default, then SIGKILL)
# Our app handles SIGTERM and shuts down within 30s
STOPSIGNAL SIGTERM

# Default command
CMD ["/app/jadevectordb"]

#############################################################################
# Build Information
#############################################################################
# Build this image with:
#   docker build -f Dockerfile.unified -t jadevectordb:latest .
#
# Run with:
#   docker run -d \
#     --name jadevectordb \
#     -p 8080:8080 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/config:/app/config \
#     jadevectordb:latest
#
# Development build with tests:
#   docker build -f Dockerfile.unified \
#     --build-arg BUILD_TESTS=ON \
#     --build-arg BUILD_BENCHMARKS=ON \
#     -t jadevectordb:dev .
#############################################################################
