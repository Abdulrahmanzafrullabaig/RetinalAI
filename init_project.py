#!/usr/bin/env python3
"""
Project Initialization Script
This script helps initialize the RetinalAI project by downloading required models
and setting up the environment.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Docker found: {result.stdout.strip()}")
            return True
        else:
            logger.error("Docker is not installed or not in PATH")
            return False
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH")
        return False

def check_docker_compose():
    """Check if Docker Compose is installed"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            # Try docker compose (newer version)
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Docker Compose found: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker Compose is not installed")
                return False
    except FileNotFoundError:
        logger.error("Docker Compose is not installed")
        return False

def download_models():
    """Download models using the backend script"""
    backend_dir = os.path.join(os.getcwd(), 'backend')
    if not os.path.exists(backend_dir):
        logger.error("Backend directory not found")
        return False
    
    try:
        logger.info("Downloading models...")
        result = subprocess.run([
            sys.executable, 
            os.path.join(backend_dir, 'download_models.py')
        ], cwd=backend_dir)
        
        if result.returncode == 0:
            logger.info("Models downloaded successfully")
            return True
        else:
            logger.error("Failed to download models")
            return False
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False

def build_docker_images():
    """Build Docker images"""
    try:
        logger.info("Building Docker images...")
        result = subprocess.run(['docker-compose', 'build'])
        
        if result.returncode == 0:
            logger.info("Docker images built successfully")
            return True
        else:
            logger.error("Failed to build Docker images")
            return False
    except Exception as e:
        logger.error(f"Error building Docker images: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("Initializing RetinalAI Project...")
    
    # Check prerequisites
    if not check_docker():
        logger.error("Please install Docker before proceeding")
        return False
    
    if not check_docker_compose():
        logger.error("Please install Docker Compose before proceeding")
        return False
    
    # Download models
    logger.info("Step 1: Downloading models...")
    if not download_models():
        logger.error("Failed to download models")
        return False
    
    # Build Docker images
    logger.info("Step 2: Building Docker images...")
    if not build_docker_images():
        logger.error("Failed to build Docker images")
        return False
    
    logger.info("Project initialization completed successfully!")
    logger.info("You can now run the project with: docker-compose up")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)