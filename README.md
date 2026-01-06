# RetinalAI - Diabetic Retinopathy Detection System

An advanced AI-powered diagnostic system that uses deep learning to analyze retinal images and detect diabetic retinopathy with high accuracy.

## Features

- **AI-Powered Analysis**: Deep learning models for accurate retinal image classification.
- **Multi-Model Ensemble**: Combines predictions from multiple state-of-the-art architectures.
- **Explainable AI (XAI)**: Grad-CAM visualizations to highlight affected areas.
- **Automated Reports**: Generates comprehensive PDF reports with diagnosis results.
- **Collaboration Platform**: Tools for doctor-patient interaction.
- **Appointment Management**: Integrated scheduling system.

## Tech Stack

### Frontend
- **Framework**: React with Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React

### Backend
- **Framework**: Flask (Python)
- **AI/ML**: PyTorch, torchvision
- **Database**: SQLite (SQLAlchemy)
- **Containerization**: Docker & Docker Compose

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM (recommended for model loading)
- Internet connection (for initial model downloads)

## Quick Start

We provide convenience scripts to quickly get the application up and running.

**For Windows:**
```cmd
start.bat
```

**For Linux/Mac:**
```bash
./start.sh
```

These scripts will automatically check for prerequisites, build the Docker images, and start the services.

## Manual Setup

If you prefer to run things manually:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd final-version-of-frontend
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## Model Downloads

The system automatically handles model dependencies. On the first run, it will download the necessary pre-trained weights from Google Drive:
- Clinical Prediction Model
- Stage 1 Fundus Classifier
- ResNet50, EfficientNet, VGG16, and DenseNet121 Models

## Project Structure

```
.
├── backend/                 # Flask backend application
│   ├── app.py              # Main application file
│   ├── requirements.txt    # Python dependencies
│   ├── download_models.py  # Script to download models
│   └── models/             # AI Model storage
├── src/                    # React frontend source code
├── Dockerfile              # Backend Docker configuration
├── Dockerfile.frontend     # Frontend Docker configuration
├── docker-compose.yml      # Container orchestration
├── start.bat               # Windows startup script
├── start.sh                # Unix startup script
└── README.md               # Project documentation
```

## Local Development

You can run the frontend and backend independently for development.

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*Runs on http://localhost:5000*

### Frontend
```bash
npm install
npm run dev
```
*Runs on http://localhost:5173 (default Vite port) or similar*

## Environment Variables

### Backend
- `FLASK_ENV`: `development` or `production`

### Frontend
- `VITE_API_URL`: URL of the backend API (defaults to `http://localhost:5000`)

## API Endpoints

- `POST /api/register` - User registration
- `POST /api/login` - User login
- `POST /api/predict` - Upload image for analysis
- `GET /api/results` - Retrieve past analysis results
- `GET /api/appointments` - Manage appointments

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is for demonstration and educational purposes.
