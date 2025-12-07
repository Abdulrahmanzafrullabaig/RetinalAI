# RetinalAI - Diabetic Retinopathy Detection System

## Overview

RetinalAI is an advanced AI-powered diagnostic system that uses deep learning to analyze retinal images and detect diabetic retinopathy with high accuracy.

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Setup**: 
   - Models are now loaded directly from Google Drive - no manual download required
   - Test that models load correctly from Google Drive:
     ```bash
     python test_gdrive_model_loading.py
     ```
   - Alternatively, you can manually download models using:
     ```bash
     python download_models.py
     ```

4. Start the backend server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

## Features

- AI-powered retinal image analysis
- Multi-model ensemble prediction
- Grad-CAM explainability for transparent diagnosis
- Automated report generation
- Doctor-patient collaboration platform
- Appointment scheduling system

## Technology Stack

- **Frontend**: React 18 with TypeScript, Vite, Tailwind CSS
- **Backend**: Python Flask
- **AI Models**: ResNet50, EfficientNet, VGG16, DenseNet121
- **Database**: SQLite.

## Usage

1. Register/Login to the system
2. Upload a retinal fundus image
3. Get AI-powered analysis of diabetic retinopathy stages
4. View detailed reports with Grad-CAM explanations
5. Share results with doctors
6. Schedule appointments with specialists

## Model Configuration

The system supports loading models from Google Drive. Refer to [MODEL_SETUP_INSTRUCTIONS.md](MODEL_SETUP_INSTRUCTIONS.md) for detailed setup instructions.
