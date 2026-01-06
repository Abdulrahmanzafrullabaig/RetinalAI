@echo off
title RetinalAI Startup Script

echo Starting RetinalAI Project...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Build and start the containers
echo Building Docker images...
docker-compose build

echo Starting containers...
docker-compose up -d

echo Containers started successfully!
echo Frontend is available at: http://localhost:3000
echo Backend API is available at: http://localhost:5000

pause