FROM python:3.13-slim

# Set working directory
WORKDIR /srv/centrifuge

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install Python dependencies using pip (uv would be installed in container)
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 centrifuge && \
    chown -R centrifuge:centrifuge /srv/centrifuge

USER centrifuge

# Default command (can be overridden)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]