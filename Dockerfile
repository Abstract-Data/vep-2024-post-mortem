# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git curl openssh-client
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pip install uv

# Set the working directory in the container
WORKDIR /app

# Copy the project files
COPY . .


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
ENV PATH="/root/.local/bin:$PATH"

# Add GitHub to known hosts
RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Use build-time secret for GitHub token
RUN --mount=type=ssh \
    bash -c 'git config --global url."git@github.com:".insteadOf "https://github.com/" && \
    uv sync --frozen --no-install-project --no-dev'
    
# Command to run the application using uvicorn
CMD ["uv", "run", "./processing.py"]