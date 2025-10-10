# JadeVectorDB Developer Onboarding Guide

Welcome to the JadeVectorDB project! This guide will walk you through setting up your local development environment to get you up and running.

## 1. Prerequisites

Before you begin, ensure you have the following tools installed on your system:

- **Git:** For version control.
- **C++ Toolchain:** A modern C++ compiler (GCC, Clang, or MSVC) that supports C++20.
- **CMake:** Version 3.15 or higher, for building the C++ backend.
- **Node.js:** Version 18 or higher, for the Next.js frontend.
- **Python:** Version 3.8 or higher, for the Python-based CLI tools.
- **Docker and Docker Compose:** For running the entire application stack locally in containers.

You can check if most of these are installed by running the provided prerequisite checker script:

```bash
sh .specify/scripts/bash/check-prerequisites.sh
```

## 2. Cloning the Repository

Start by cloning the repository to your local machine:

```bash
git clone <repository-url>
cd JadeVectorDB
```

## 3. Backend Setup (C++)

The backend consists of C++ microservices. It is managed by CMake.

1.  **Install Dependencies:** The project uses several third-party libraries (Eigen, OpenBLAS, FlatBuffers, gRPC, etc.). These are managed via CMake's `FetchContent` or are expected to be available on your system. The build script will handle them.

2.  **Build the Backend:**

    ```bash
    # Navigate to the backend directory
    cd backend

    # Configure the project with CMake
    cmake -B build

    # Build the project
    cmake --build build
    ```

    This will compile the backend services and place the executables in the `backend/build/` directory.

## 4. Frontend Setup (Next.js)

The frontend is a Next.js web application.

1.  **Navigate to the frontend directory:**

    ```bash
    cd frontend
    ```

2.  **Install Dependencies:**

    ```bash
    npm install
    ```

    This will download all the required Node.js packages.

## 5. CLI Setup (Python)

The Python CLI allows for easy interaction with the database from the command line.

1.  **Navigate to the Python CLI directory:**

    ```bash
    cd cli/python
    ```

2.  **Install in Editable Mode:** It's recommended to install the package in editable mode so that your changes are immediately reflected.

    ```bash
    pip install -e .
    ```

## 6. Running the System Locally with Docker Compose

The easiest way to run the entire stack (backend, frontend, and dependencies) is by using the provided Docker Compose configuration.

1.  **Ensure Docker is running.**

2.  **From the project root directory, run:**

    ```bash
    docker-compose up --build
    ```

    The `--build` flag ensures that the Docker images are rebuilt to reflect any code changes you've made. This command will start all the services defined in `docker-compose.yml`.

3.  **Accessing the Services:**
    -   **Web UI:** Open your browser and navigate to `http://localhost:3000`.
    -   **API:** The API will be accessible at `http://localhost:8080` (or the port configured in your environment).

## 7. Development Workflow

- **Making Changes:** Make your code changes in the `backend`, `frontend`, or `cli` directories.
- **Testing (Backend):** To run the C++ tests, execute `ctest` from the `backend/build` directory.
- **Rebuilding:** If you are not using Docker, you will need to manually rebuild the specific component you are working on.
- **Running with Docker:** If you are using Docker, simply run `docker-compose up --build` again to restart the system with your latest changes.
