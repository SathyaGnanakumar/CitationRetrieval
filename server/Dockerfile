# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY evaluate.py ./

# Install dependencies using uv
RUN uv sync

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/graphs /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATASET_DIR=/app/data/scholar_copilot_eval_data_1k.json
ENV GRAPH_OUTPUT_DIR=/app/graphs

# Expose port if needed for future API
EXPOSE 8000

# Default command
CMD ["uv", "run", "python", "-c", "print('Citation Retrieval System - Ready to use')"]
