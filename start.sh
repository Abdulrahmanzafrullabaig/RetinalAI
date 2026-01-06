#!/bin/bash

# RetinalAI Startup Script

echo "Starting RetinalAI Project..."

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build and start the containers
echo "Building Docker images..."
docker-compose build

echo "Starting containers..."
docker-compose up -d

echo "Containers started successfully!"
echo "Frontend is available at: http://localhost:3000"
echo "Backend API is available at: http://localhost:5000"