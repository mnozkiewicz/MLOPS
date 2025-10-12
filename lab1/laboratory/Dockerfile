# Dockerfile

# Use a minimal Python image as the base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Set the working directory in the container
WORKDIR /app

# Copy only dependency files first (to leverage caching)
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv
RUN uv sync

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Run the application with uv
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]