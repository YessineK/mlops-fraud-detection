FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy all backend files
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Set Python path and run
ENV PYTHONPATH=/app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
