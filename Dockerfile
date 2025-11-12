# Citation Retrieval Project - Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY .env.example .env.example

# Copy source code
COPY evaluation/ evaluation/
COPY baselines/ baselines/
COPY datasets/ datasets/
COPY main.py .
COPY visualization.py .

# Install dependencies using uv
RUN uv sync

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["uv", "run", "python", "-c", "print('Citation Retrieval Docker Container Ready! Run: uv run main.py')"]
