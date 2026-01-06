# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for downloading models from Google Drive
RUN pip install --no-cache-dir gdown

# Copy backend code
COPY backend/ ./backend

# Create models directory
RUN mkdir -p backend/models

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=backend/app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the download script to get models
RUN cd backend && python download_models.py

# Run the application
CMD ["python", "backend/app.py"]