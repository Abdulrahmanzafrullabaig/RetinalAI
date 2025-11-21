# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for Google Drive downloads
RUN pip install gdown

# Copy backend code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]