# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git

# Set environment variables
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

# Set the working directory in the container
# Copy the project files
COPY . /app
WORKDIR /app


# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# # Copy from the cache instead of linking since it's a mounted volume
# ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
# RUN pip install uv
# ADD . /app
# RUN uv sync --frozen --no-install-project --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN=$(cat /run/secrets/github_token)

# Install the project's dependencies using the lockfile and settings
RUN pip install uv
RUN uv sync --frozen --no-install-project --no-dev
# # Copy the rest of the application code into the container
# COPY . .

# Command to run the application using uvicorn
CMD ["uv", "run", "./processing.py"]