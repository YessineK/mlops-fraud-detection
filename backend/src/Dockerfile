FROM python:3.12-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# ✅ IMPORTANT - Copier le modèle au bon endroit
COPY src/processors /app/processors

# Expose port
EXPOSE 8000

# Run from src folder
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
