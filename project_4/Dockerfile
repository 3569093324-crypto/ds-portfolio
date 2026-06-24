# Project 4: User Repurchase Prediction
# docker build -t repurchase-model .
# docker run repurchase-model

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Create output directories
RUN mkdir -p visuals output

# Default command: run model comparison
CMD ["python", "model_comparison.py"]
