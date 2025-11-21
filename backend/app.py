from flask import Flask, request, redirect, url_for, flash, session, jsonify, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import sqlite3
import uuid
from functools import wraps
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
import timm  # For mobilenetv4_conv_large
from typing import Tuple, Optional, Dict, Any
import torch.nn as nn
import torchvision.models as models
import logging
import re
try:
    import google.generativeai as genai
except ImportError:
    # Define logger first before using it
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")
    genai = None
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = 'AIzaSyBLmnTrOyo-orYEJGoJYzJD_gvzhnmHOK4'
if genai:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS for frontend communication
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_DIR'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_SECURE'] = False # Changed to False for local testing, set to True in production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/reports', exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

# DR Classification stages
DR_STAGES = {
    0: "No DR - No visible abnormalities",
    1: "Mild DR - Microaneurysms only",
    2: "Moderate DR - More than microaneurysms",
    3: "Severe DR - Extensive abnormalities",
    4: "Proliferative DR - Neovascularization present"
}
CLASSES = list(DR_STAGES.values())
MODEL_ARCHITECTURES = ['resnet50', 'efficientnet', 'vgg16', 'densenet121']

# Stage 1 model for fundus/non-fundus detection
stage1_model = None  # Global variable for stage 1 model

# Comprehensive Medical Knowledge Base for Chatbot
KNOWLEDGE_BASE = {
    'product_info': {
        'retinalai': {
            'description': 'RetinalAI is an advanced AI-powered diagnostic system that uses deep learning to analyze retinal images and detect diabetic retinopathy with high accuracy.',
            'features': [
                'AI-powered retinal image analysis',
                'Multi-model ensemble prediction',
                'Grad-CAM explainability for transparent diagnosis',
                'Automated report generation',
                'Doctor-patient collaboration platform',
                'Appointment scheduling system'
            ],
            'accuracy': '95%+ accuracy across multiple validation datasets',
            'models': 'Uses ensemble of ResNet50, EfficientNet, VGG16, and DenseNet121 architectures'
        }
    },
    'diseases': {
        'diabetic_retinopathy': {
            'definition': 'Diabetic retinopathy is a diabetes complication that affects the eyes. It is caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).',
            'causes': [
                'High blood sugar levels over time',
                'High blood pressure',
                'High cholesterol',
                'Pregnancy',
                'Tobacco use',
                'Duration of diabetes'
            ],
            'symptoms': {
                'early': ['Often no symptoms', 'Mild vision changes'],
                'advanced': [
                    'Spots or dark strings floating in vision (floaters)',
                    'Blurred vision',
                    'Fluctuating vision',
                    'Dark or empty areas in vision',
                    'Vision loss'
                ]
            },
            'risk_factors': [
                'Poor blood sugar control',
                'High blood pressure',
                'High cholesterol',
                'Pregnancy',
                'Tobacco use',
                'Race (higher risk in African Americans, Hispanics, Native Americans)',
                'Duration of diabetes'
            ]
        }
    },
    'stages': {
        'diabetic_retinopathy_stages': {
            'no_dr': {
                'stage': 0,
                'name': 'No Diabetic Retinopathy',
                'description': 'No visible abnormalities or signs of diabetic retinopathy',
                'characteristics': ['Normal retinal blood vessels', 'No microaneurysms', 'No hemorrhages'],
                'action': 'Continue regular monitoring and maintain good diabetes control'
            },
            'mild_dr': {
                'stage': 1,
                'name': 'Mild Nonproliferative Diabetic Retinopathy',
                'description': 'Microaneurysms only - small areas of balloon-like swelling in retinal blood vessels',
                'characteristics': ['Microaneurysms present', 'No other abnormalities'],
                'action': 'Annual eye exams and improved diabetes management'
            },
            'moderate_dr': {
                'stage': 2,
                'name': 'Moderate Nonproliferative Diabetic Retinopathy',
                'description': 'More than just microaneurysms but less than severe NPDR',
                'characteristics': ['Microaneurysms', 'Small hemorrhages', 'Hard exudates', 'Cotton wool spots'],
                'action': 'More frequent monitoring (every 6-12 months) and diabetes optimization'
            },
            'severe_dr': {
                'stage': 3,
                'name': 'Severe Nonproliferative Diabetic Retinopathy',
                'description': 'Extensive retinal abnormalities without neovascularization',
                'characteristics': [
                    'Extensive hemorrhages and microaneurysms',
                    'Venous abnormalities',
                    'Intraretinal microvascular abnormalities'
                ],
                'action': 'Urgent ophthalmologic referral and close monitoring every 3-4 months'
            },
            'proliferative_dr': {
                'stage': 4,
                'name': 'Proliferative Diabetic Retinopathy',
                'description': 'Advanced stage with new blood vessel growth (neovascularization)',
                'characteristics': [
                    'Neovascularization of disc or elsewhere',
                    'Vitreous hemorrhage',
                    'Retinal detachment risk'
                ],
                'action': 'Immediate treatment required - laser therapy, injections, or surgery'
            }
        }
    },
    'prevention': {
        'primary_prevention': [
            'Maintain excellent blood sugar control (HbA1c < 7%)',
            'Control blood pressure (< 140/90 mmHg)',
            'Manage cholesterol levels',
            'Regular exercise and healthy diet',
            'Avoid smoking and excessive alcohol',
            'Take prescribed medications consistently'
        ],
        'screening': [
            'Annual comprehensive eye exams for all diabetics',
            'More frequent exams if retinopathy is present',
            'Immediate exam if vision changes occur',
            'Regular monitoring during pregnancy'
        ],
        'lifestyle_modifications': [
            'Mediterranean or DASH diet',
            'Regular aerobic exercise (150 minutes/week)',
            'Weight management',
            'Stress reduction techniques',
            'Adequate sleep (7-9 hours)',
            'UV protection for eyes'
        ]
    },
    'treatment': {
        'medical_management': [
            'Optimize diabetes control',
            'Blood pressure management',
            'Cholesterol control',
            'Anti-VEGF injections for macular edema',
            'Steroid injections in some cases'
        ],
        'surgical_treatments': [
            'Laser photocoagulation',
            'Vitrectomy for severe cases',
            'Retinal detachment repair',
            'Cataract surgery if needed'
        ],
        'follow_up': [
            'Regular ophthalmologic monitoring',
            'Diabetes management team coordination',
            'Patient education and support',
            'Vision rehabilitation if needed'
        ]
    },
    'suggestions': {
        'immediate_actions': {
            'no_dr': [
                'Continue current diabetes management',
                'Schedule annual eye exams',
                'Maintain healthy lifestyle',
                'Monitor blood sugar regularly'
            ],
            'mild_moderate': [
                'Improve diabetes control',
                'Schedule more frequent eye exams',
                'Consider diabetes education classes',
                'Discuss treatment options with endocrinologist'
            ],
            'severe_proliferative': [
                'Seek immediate ophthalmologic care',
                'Optimize diabetes management urgently',
                'Consider specialist referrals',
                'Prepare for potential treatments'
            ]
        },
        'long_term_care': [
            'Build a healthcare team (endocrinologist, ophthalmologist, primary care)',
            'Use diabetes management apps and tools',
            'Join diabetes support groups',
            'Stay informed about new treatments',
            'Maintain regular follow-up appointments',
            'Keep detailed health records'
        ]
    }
}

# Model names and file mappings
models_dict = {}  # Global dictionary to store models and conv layers
clinical_model = None  # Global variable for clinical data model
model_names = ['resnet50', 'efficientnet', 'vgg16', 'densenet121']
model_files = {
    'resnet50': 'model1.pth',
    'efficientnet': 'model2.pth',
    'vgg16': 'model3.pth',
    'densenet121': 'model4.pth'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def is_fundus_image(image_path: str) -> bool:
    """Check if the uploaded image is likely a fundus image based on visual characteristics"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Check if image is roughly circular (fundus images are typically circular)
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use HoughCircles to detect circular structures
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=min(image.width, image.height) // 2
        )
        
        # Check for circular structure (fundus images typically have a circular boundary)
        has_circular_structure = circles is not None and len(circles[0]) > 0
        
        # Check color characteristics - fundus images typically have reddish/orange tones
        # Calculate average color in HSV space
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Focus on the center region where the optic disc might be
        h, w = hsv.shape[:2]
        center_region = hsv[h//4:3*h//4, w//4:3*w//4]
        
        # Check if there are reddish/orange hues typical of retinal images
        hue_values = center_region[:, :, 0]
        # Fundus images typically have hues in the red-orange range (0-30 or 160-180 in HSV)
        red_orange_pixels = np.sum((hue_values <= 30) | (hue_values >= 160))
        total_pixels = hue_values.size
        red_orange_ratio = red_orange_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check saturation - fundus images typically have moderate to high saturation
        saturation_values = center_region[:, :, 1]
        avg_saturation = float(saturation_values.mean())
        
        # Fundus image criteria:
        # 1. Has circular structure OR good red/orange color characteristics
        # 2. Reasonable saturation levels
        # 3. Image should not be too bright or too dark (medical imaging quality)
        
        # Check brightness - fundus images have specific brightness characteristics
        value_values = center_region[:, :, 2]  # V channel in HSV
        avg_brightness = float(value_values.mean())
        
        is_fundus = (
            (has_circular_structure or red_orange_ratio > 0.2) and 
            avg_saturation > 25 and  # Minimum saturation threshold
            50 < avg_brightness < 200  # Brightness range for medical images
        )
        
        logger.info(f"Image validation - Circular: {has_circular_structure}, Red/Orange ratio: {red_orange_ratio:.3f}, Avg saturation: {avg_saturation:.1f}, Avg brightness: {avg_brightness:.1f}, Is fundus: {is_fundus}")
        return bool(is_fundus)
        
    except Exception as e:
        logger.error(f"Error validating fundus image: {e}")
        # If validation fails, assume it's not a fundus image for safety
        return False

def get_stage1_transforms() -> transforms.Compose:
    """Get image transforms for stage 1 model (InceptionV3)"""
    return transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_stage1_fundus(image_path: str) -> Tuple[bool, float, Optional[str]]:
    """Predict if image is fundus using stage 1 model
    
    Returns:
        Tuple[bool, float, Optional[str]]: (is_fundus, confidence, error_message)
    """
    if stage1_model is None:
        # Fallback to rule-based validation
        logger.info("Stage 1 model not available, using rule-based fundus validation")
        is_fundus = is_fundus_image(image_path)
        return is_fundus, 0.8 if is_fundus else 0.2, None
    
    try:
        # Preprocess image for InceptionV3
        transform = get_stage1_transforms()
        image = Image.open(image_path).convert('RGB')
        tensor_image = transform(image)
        if isinstance(tensor_image, torch.Tensor):
            tensor_image = tensor_image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = stage1_model(tensor_image)

            # Apply softmax to get probabilities for 2-class classification
            probabilities = torch.softmax(output, dim=1)
            # Assuming class 1 is 'fundus' and class 0 is 'non-fundus'
            fundus_probability = float(probabilities[0][1].item())
            non_fundus_probability = float(probabilities[0][0].item())
            
            # Use fundus probability for decision
            threshold = 0.5
            is_fundus = fundus_probability > threshold
            confidence = fundus_probability if is_fundus else non_fundus_probability
            
            logger.info(f"Stage 1 prediction - Is fundus: {is_fundus}, Fundus probability: {fundus_probability:.3f}, Confidence: {confidence:.3f}")
            return is_fundus, confidence, None
            
    except Exception as e:
        logger.error(f"Stage 1 prediction failed: {e}")
        # Fallback to rule-based validation
        is_fundus = is_fundus_image(image_path)
        return is_fundus, 0.5, f"Stage 1 model error: {str(e)}"

def initialize_model(arch: str) -> Tuple[nn.Module, nn.Module]:
    """Initialize an empty model with correct architecture"""
    logger.info(f"Initializing model: {arch}")
    try:
        if arch == 'resnet50':
            model = torchvision_models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
            final_conv_layer = model.layer4[-1].conv3
        elif arch == 'efficientnet':
            model = torchvision_models.efficientnet_b0(weights=None)
            # Get the classifier layer properly
            classifier = model.classifier
            try:
                if hasattr(classifier, '__getitem__') and hasattr(classifier, '__len__'):
                    in_features = getattr(classifier[1], 'in_features', 1280)
                    classifier[1] = nn.Linear(in_features, len(CLASSES))
            except (IndexError, AttributeError):
                pass
            # Get features properly - use a safe approach
            try:
                features = model.features
                # Use a more conservative approach for getting the final conv layer
                final_conv_layer = features
                for layer in reversed(list(features.children()) if hasattr(features, 'children') else []):
                    if hasattr(layer, 'weight'):
                        final_conv_layer = layer
                        break
            except (IndexError, AttributeError):
                final_conv_layer = model.features
        elif arch == 'vgg16':
            model = torchvision_models.vgg16(weights=None)
            # Get the classifier layer properly
            classifier = model.classifier
            try:
                if hasattr(classifier, '__getitem__') and hasattr(classifier, '__len__'):
                    in_features = getattr(classifier[6], 'in_features', 4096)
                    classifier[6] = nn.Linear(in_features, len(CLASSES))
            except (IndexError, AttributeError):
                pass
            # Get features properly - use a safe approach
            try:
                features = model.features
                # Use a more conservative approach
                final_conv_layer = features
                feature_list = list(features.children()) if hasattr(features, 'children') else []
                if len(feature_list) >= 3:
                    final_conv_layer = feature_list[-3]
            except (IndexError, AttributeError):
                final_conv_layer = model.features
        elif arch == 'densenet121':
            # This is a DenseNet121 model
            model = models.densenet121(pretrained=False)  # pretrained=False since we're loading custom weights
            # Modify the final layer for our number of classes
            model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
            # For DenseNet121, we need to find the appropriate layer for Grad-CAM
            # The final dense block is typically used for Grad-CAM
            final_conv_layer = model.features.norm5
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Cast final_conv_layer to nn.Module for type safety
        if not isinstance(final_conv_layer, nn.Module):
            final_conv_layer = model  # Fallback to the entire model
            
        return model, final_conv_layer
    except Exception as e:
        logger.error(f"Model initialization failed for {arch}: {e}")
        raise RuntimeError(f"Model initialization failed for {arch}: {str(e)}")

def init_db():
    """Initialize the database"""
    conn = None
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 1000000")
        conn.execute("PRAGMA temp_store = MEMORY")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      role TEXT NOT NULL,
                      full_name TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS reports
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      filename TEXT NOT NULL,
                      predictions TEXT NOT NULL,
                      final_prediction INTEGER NOT NULL,
                      confidence REAL NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      doctor_notes TEXT DEFAULT '',
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS appointments
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      patient_id INTEGER NOT NULL,
                      doctor_id INTEGER NOT NULL,
                      appointment_date TEXT NOT NULL,
                      appointment_time TEXT NOT NULL,
                      reason TEXT,
                      status TEXT DEFAULT 'pending',
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (patient_id) REFERENCES users (id),
                      FOREIGN KEY (doctor_id) REFERENCES users (id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS shared_reports
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      report_id INTEGER NOT NULL,
                      doctor_id INTEGER NOT NULL,
                      shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      status TEXT DEFAULT 'new', -- new, reviewed, urgent
                      FOREIGN KEY (report_id) REFERENCES reports (id),
                      FOREIGN KEY (doctor_id) REFERENCES users (id))''')
        conn.commit()
        logger.info("Database initialized successfully with WAL mode")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def load_stage1_model() -> Optional[nn.Module]:
    """Load stage 1 model for fundus/non-fundus detection"""
    # Google Drive file ID for fundus classifier
    file_id = "1sWu9cAiz7Z2DqnEsXllpqOcMToBDF2Z7"
    
    try:
        import gdown
        import tempfile
        import os
        
        # Create a temporary file to download the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Construct Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"
            logger.info("Downloading stage 1 model (fundus classifier) from Google Drive...")
            
            # Download the model file directly to temporary file
            gdown.download(url, temp_path, quiet=False)
            
            # Check if download was successful
            if not os.path.exists(temp_path):
                logger.error("Failed to download stage 1 model from Google Drive")
                return None
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                logger.error("Downloaded stage 1 model file is empty")
                return None
            
            logger.info(f"Successfully downloaded stage 1 model ({file_size} bytes)")
            
            # Initialize the InceptionV3 model for binary classification (2 classes)
            model = models.inception_v3(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification: fundus vs non-fundus
            model.aux_logits = False  # Disable auxiliary logits
            
            # Load the checkpoint
            checkpoint = torch.load(temp_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Handle potential DataParallel wrapper
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            model.to(device).eval()
            
            # Test the model
            test_input = torch.rand(1, 3, 299, 299).to(device)
            with torch.no_grad():
                output = model(test_input)
                # Check if output is the expected shape for binary classification (2 classes)
                if hasattr(output, 'shape') and len(output.shape) > 1 and output.shape[1] != 2:
                    logger.error(f"Stage 1 model output shape mismatch: expected 2, got {output.shape[1] if len(output.shape) > 1 else 'unknown'}")
                    return None
            
            logger.info("Stage 1 FundusClassifier model loaded successfully from Google Drive")
            return model
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except ImportError:
        logger.warning("gdown library not installed. Please install with: pip install gdown")
        return None
    except Exception as e:
        logger.error(f"Error loading stage 1 model from Google Drive: {e}")
        return None

def load_model(arch: str, path: str) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[str]]:
    """Load model weights from file with error handling"""
    if not os.path.exists(path):
        logger.error(f"Model file not found at {path}")
        return None, None, f"Model file not found at {path}"

    try:
        model, final_conv_layer = initialize_model(arch)
        state_dict = torch.load(path, map_location=device)
        if arch == 'mobilenetv4_conv_large' and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        # Test input to ensure model can be run
        if arch == 'mobilenetv4_conv_large':
            test_input = torch.rand(1, 3, 384, 384).to(device)
        else:
            test_input = torch.rand(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
            if output.shape[1] != len(CLASSES):
                logger.error(f"Model output shape mismatch for {arch}")
                return None, None, f"Model output shape mismatch for {arch}"
        logger.info(f"Model {arch} loaded successfully from {path}")
        return model, final_conv_layer, None
    except Exception as e:
        logger.error(f"Error loading {arch}: {e}")
        return None, None, f"Error loading {arch}: {str(e)}"

def load_model_from_gdrive(arch: str, file_id: str) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[str]]:
    """Load model weights directly from Google Drive with error handling"""
    try:
        import gdown
        import tempfile
        import os
        
        # Create a temporary file to download the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Construct Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"
            logger.info(f"Downloading {arch} model from Google Drive...")
            
            # Download the model file directly to temporary file
            gdown.download(url, temp_path, quiet=False)
            
            # Check if download was successful
            if not os.path.exists(temp_path):
                logger.error(f"Failed to download {arch} model from Google Drive")
                return None, None, f"Failed to download {arch} model from Google Drive"
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                logger.error(f"Downloaded {arch} model file is empty")
                return None, None, f"Downloaded {arch} model file is empty"
            
            logger.info(f"Successfully downloaded {arch} model ({file_size} bytes)")
            
            # Initialize model architecture
            model, final_conv_layer = initialize_model(arch)
            
            # Load state dict from temporary file
            state_dict = torch.load(temp_path, map_location=device)
            
            # Handle DataParallel wrapper if present
            if arch == 'mobilenetv4_conv_large' and any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load the state dict into the model
            model.load_state_dict(state_dict, strict=False)
            model.to(device).eval()
            
            # Test input to ensure model can be run
            if arch == 'mobilenetv4_conv_large':
                test_input = torch.rand(1, 3, 384, 384).to(device)
            else:
                test_input = torch.rand(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(test_input)
                if output.shape[1] != len(CLASSES):
                    logger.error(f"Model output shape mismatch for {arch}")
                    return None, None, f"Model output shape mismatch for {arch}"
            
            logger.info(f"Model {arch} loaded successfully from Google Drive")
            return model, final_conv_layer, None
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except ImportError:
        logger.error("gdown library not installed. Please install with: pip install gdown")
        return None, None, "gdown library not installed. Please install with: pip install gdown"
    except Exception as e:
        logger.error(f"Error loading {arch} from Google Drive: {e}")
        return None, None, f"Error loading {arch} from Google Drive: {str(e)}"

def load_clinical_model() -> Optional[nn.Module]:
    """Load clinical data model for tabular prediction"""
    # Google Drive file ID for clinical model
    file_id = "18tJ7d4BbdWReVUCWSvfF_cY3yMh46PZI"
    
    try:
        import gdown
        import tempfile
        import os
        
        # Create a temporary file to download the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Construct Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"
            logger.info("Downloading clinical model from Google Drive...")
            
            # Download the model file directly to temporary file
            gdown.download(url, temp_path, quiet=False)
            
            # Check if download was successful
            if not os.path.exists(temp_path):
                logger.error("Failed to download clinical model from Google Drive")
                return None
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                logger.error("Downloaded clinical model file is empty")
                return None
            
            logger.info(f"Successfully downloaded clinical model ({file_size} bytes)")
            
            # Define the model class (must match the training code)
            class DRClassifier(nn.Module):
                def __init__(self, input_size=12):  # 12 features: Age, Gender, Blood_Pressure, Diabetes, Liver_Problem, Kidney_Problem, Nerves_Problem, Heart_Problem, Feet_Problem, Skin_Problem, Smoking, HbA1c
                    super(DRClassifier, self).__init__()
                    self.fc1 = nn.Linear(input_size, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 1)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.sigmoid(self.fc3(x))
                    return x
            
            # Load the trained model
            model = DRClassifier(12).to(device)
            model.load_state_dict(torch.load(temp_path, map_location=device))
            model.eval()
            
            logger.info("Clinical model loaded successfully from Google Drive")
            return model
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except ImportError:
        logger.error("gdown library not installed. Please install with: pip install gdown")
        return None
    except Exception as e:
        logger.error(f"Error loading clinical model from Google Drive: {e}")
        return None

def process_clinical_data(data: Dict) -> torch.Tensor:
    """Process and encode clinical data for model input (12 features)"""
    try:
        # Extract features in the correct order for the new model
        # Features: Age, Gender, Blood_Pressure, Diabetes, Liver_Problem, Kidney_Problem, 
        # Nerves_Problem, Heart_Problem, Feet_Problem, Skin_Problem, Smoking, HbA1c
        age = float(data.get('Age', 0))
        gender = 1 if data.get('Gender') == 'Male' else 0
        
        # Blood pressure encoding
        bp_mapping = {
            'Normal': 0, 
            'Prehypertension': 1, 
            'Hypertension Stage 1': 2, 
            'Hypertension Stage 2': 3
        }
        blood_pressure = bp_mapping.get(data.get('Blood_Pressure', 'Normal'), 0)
        
        # Comorbidities (binary features)
        diabetes = 1 if data.get('Diabetes', 'No') == 'Yes' else 0
        liver_problem = 1 if data.get('Liver_Problem', 'No') == 'Yes' else 0
        kidney_problem = 1 if data.get('Kidney_Problem', 'No') == 'Yes' else 0
        nerves_problem = 1 if data.get('Nerves_Problem', 'No') == 'Yes' else 0
        heart_problem = 1 if data.get('Heart_Problem', 'No') == 'Yes' else 0
        feet_problem = 1 if data.get('Feet_Problem', 'No') == 'Yes' else 0
        skin_problem = 1 if data.get('Skin_Problem', 'No') == 'Yes' else 0
        smoking = 1 if data.get('Smoking', 'No') == 'Yes' else 0
        hba1c = float(data.get('HbA1c', 0))
        
        # Create feature vector (12 features total)
        features = torch.tensor([
            age, gender, blood_pressure, diabetes, liver_problem, 
            kidney_problem, nerves_problem, heart_problem, 
            feet_problem, skin_problem, smoking, hba1c
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        logger.info(f"Processed clinical features: age={age}, gender={gender}, bp={blood_pressure}, diabetes={diabetes}, hba1c={hba1c}")
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing clinical data: {e}")
        raise ValueError(f"Invalid clinical data: {str(e)}")

def predict_with_clinical_model(data: Dict) -> Dict:
    """Make prediction using clinical data model"""
    if clinical_model is None:
        return {'error': 'Clinical model not available'}
    
    try:
        features = process_clinical_data(data)
        
        with torch.no_grad():
            output = clinical_model(features)
            # For the new model with sigmoid activation, output is a single probability value
            probability = output.item()
            predicted_class = 1 if probability > 0.5 else 0
            confidence = probability if predicted_class == 1 else (1 - probability)
            
            prediction_text = "Positive for Diabetic Retinopathy" if predicted_class == 1 else "Negative for Diabetic Retinopathy"
            
            return {
                'predicted_class': predicted_class,
                'prediction_text': prediction_text,
                'confidence': f"{confidence:.2%}",
                'probabilities': {
                    'negative': f"{1 - probability:.2%}",
                    'positive': f"{probability:.2%}"
                }
            }
            
    except Exception as e:
        logger.error(f"Clinical prediction failed: {e}")
        return {'error': f'Prediction failed: {str(e)}'}

def load_all_models() -> Dict[str, Tuple[Any, Any, Optional[str]]]:
    """Load all models with error tracking"""
    # Load stage 1 model (fundus/non-fundus detection)
    global stage1_model
    stage1_model = load_stage1_model()
    if stage1_model is None:
        logger.error("Stage 1 model not loaded - fundus validation will not work")
    
    # Load clinical model
    global clinical_model
    clinical_model = load_clinical_model()
    if clinical_model is None:
        logger.error("Clinical model not loaded - clinical predictions will not be available")
    
    # Google Drive file IDs for models
    model_files_gdrive = {
        'resnet50': '1gFACSGC_7PON0xZoII6PJ5I9duk9ePGs',
        'efficientnet': '12codw5Zjq-ShLuhd_CUE_ZWf8DKUsRd0',
        'vgg16': '1MhQVHXSKMOTWItuau-Idjn4aSNQhACQJ',
        'densenet121': '1TPTcVIZx0NPmHypNuoIMCop8kkgxQgZZ'
    }

    loaded_models = {}
    errors = []

    for arch in MODEL_ARCHITECTURES:
        model, conv_layer, error = load_model_from_gdrive(arch, model_files_gdrive.get(arch, ''))
        loaded_models[arch] = (model, conv_layer)
        if error:
            errors.append(f"{arch}: {error}")

    if errors:
        logger.error("Model loading errors:\n" + "\n".join(errors))
    else:
        logger.info("All models loaded successfully from Google Drive.")

    return loaded_models

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def role_required(required_role: str):
    """Decorator to require a specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': 'Authentication required.'}), 401
            if session.get('role') != required_role:
                return jsonify({'success': False, 'message': f'Access denied. {required_role.capitalize()} privileges required.'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_transforms(arch: str) -> transforms.Compose:
    """Get image transforms based on architecture"""
    # DenseNet121 and other models use 224x224, MobileNetV4 uses 384x384
    size = (384, 384) if arch == 'mobilenetv4_conv_large' else (224, 224)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path: str, arch: str) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess image for model prediction"""
    transform = get_transforms(arch)
    try:
        image = Image.open(image_path).convert('RGB')
        tensor_image = transform(image)
        # Ensure we have a tensor and add batch dimension
        if isinstance(tensor_image, torch.Tensor) and hasattr(tensor_image, 'unsqueeze'):
            tensor_image = tensor_image.unsqueeze(0)
        elif not isinstance(tensor_image, torch.Tensor):
            # Fallback - convert to tensor first
            tensor_image = torch.tensor(tensor_image, dtype=torch.float32).unsqueeze(0)
        return tensor_image, image
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def predict_with_models(image_path: str) -> Dict[str, Dict]:
    """Get predictions from all models"""
    predictions = {}

    for arch in MODEL_ARCHITECTURES:
        model_data = models_dict.get(arch)
        if not model_data or len(model_data) < 2 or model_data[0] is None:
            predictions[arch] = {'error': f'Model {arch} not available or failed to load'}
            continue

        model, _ = model_data[:2]  # Only take first two elements
        try:
            image_tensor, _ = preprocess_image(image_path, arch)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                predicted_class = int(predicted_class)  # Ensure it's an int
                confidence = float(probabilities[predicted_class].item())  # Ensure it's a float
                predictions[arch] = {
                    'probabilities': probabilities.cpu().numpy().tolist(),
                    'predicted_class': predicted_class,
                    'prediction_text': DR_STAGES.get(predicted_class, 'Unknown'),
                    'confidence': f"{confidence:.2%}"
                }
        except Exception as e:
            logger.error(f"Prediction failed for {arch}: {e}")
            predictions[arch] = {'error': f'Prediction failed: {str(e)}'}

    return predictions

def majority_voting(predictions: Dict[str, Dict]) -> Tuple[int, float, Dict]:
    """Apply majority voting to ensemble predictions"""
    votes = []
    confidences = []

    for model_name, pred in predictions.items():
        if 'error' not in pred:
            votes.append(pred['predicted_class'])
            # Parse confidence from string if needed, or ensure it's a float
            confidences.append(float(pred['confidence'].replace('%','')) / 100 if isinstance(pred['confidence'], str) else pred['confidence'])

    if not votes:
        logger.warning("No valid predictions for majority voting")
        return 0, 0.0, {}

    vote_counts = {}
    for vote in votes:
        vote_counts[vote] = vote_counts.get(vote, 0) + 1

    final_prediction = max(vote_counts.items(), key=lambda x: x[1])[0]
    ensemble_confidence = float(np.mean(confidences)) if confidences else 0.0

    logger.info(f"Majority voting result: class {final_prediction}, confidence {ensemble_confidence:.2%}")
    return final_prediction, ensemble_confidence, vote_counts

def generate_gradcam_explanation(image_path: str, arch: str, target_class: int) -> Optional[str]:
    """Generate Grad-CAM explanation and return base64 encoded image string"""
    # DenseNet121 and other models use 224x224, MobileNetV4 uses 384x384
    size = (384, 384) if arch == 'mobilenetv4_conv_large' else (224, 224)
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize(size)
        image_array = np.array(image_resized) / 255.0  # Normalize to [0, 1]
        
        model_data = models_dict.get(arch)
        if not model_data or model_data[0] is None:
            logger.error(f"No valid model data for Grad-CAM ({arch})")
            return None

        model, target_layer = model_data[:2]  # Only take first two elements
        model.eval()
        model.to(device)
        
        # Prepare input tensor
        input_tensor, _ = preprocess_image(image_path, arch)
        input_tensor = input_tensor.to(device)
        
        # Create Grad-CAM object
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        # Generate class activation map
        from typing import List, Any
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets: List[Any] = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)
        
        # Get the cam for the first (and only) image in batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Create visualization by overlaying heatmap on original image
        visualization = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
        
        # Convert to BGR for OpenCV encoding
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', visualization_bgr)
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.info(f"Grad-CAM explanation generated for {arch}")
        return gradcam_b64
        
    except Exception as e:
        logger.error(f"Grad-CAM failed for {arch}: {e}")
        return None

def generate_all_gradcam_explanations(image_path: str, target_class: int) -> Dict[str, Optional[str]]:
    """Generate Grad-CAM explanations for all available models"""
    gradcam_explanations = {}
    
    for arch in MODEL_ARCHITECTURES:
        model_data = models_dict.get(arch)
        if not model_data or model_data[0] is None:
            gradcam_explanations[arch] = None
            continue
            
        gradcam_b64 = generate_gradcam_explanation(image_path, arch, target_class)
        gradcam_explanations[arch] = gradcam_b64
    
    return gradcam_explanations



def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(user_input, knowledge_base, threshold=0.3):
    """Find the best matching response from knowledge base"""
    user_input = user_input.lower().strip()
    
    # Define keywords for different categories
    keywords = {
        'product': ['retinalai', 'product', 'system', 'ai', 'accuracy', 'models', 'features', 'how does', 'what is retinalai'],
        'disease': ['diabetic retinopathy', 'diabetes', 'retinopathy', 'what is diabetic', 'causes', 'symptoms', 'risk factors'],
        'stages': ['stages', 'stage', 'mild', 'moderate', 'severe', 'proliferative', 'no dr', 'classification'],
        'prevention': ['prevention', 'prevent', 'avoid', 'screening', 'lifestyle', 'exercise', 'diet', 'control'],
        'treatment': ['treatment', 'therapy', 'surgery', 'injection', 'laser', 'cure', 'medication'],
        'suggestions': ['suggestions', 'advice', 'recommendations', 'what should i do', 'help', 'guidance']
    }
    
    # Check for keyword matches
    for category, terms in keywords.items():
        for term in terms:
            if term in user_input:
                if category == 'product':
                    return 'product_info', knowledge_base.get('product_info', {}).get('retinalai', {})
                elif category == 'disease':
                    return 'disease_info', knowledge_base.get('diseases', {}).get('diabetic_retinopathy', {})
                elif category == 'stages':
                    return 'stages_info', knowledge_base.get('stages', {}).get('diabetic_retinopathy_stages', {})
                elif category == 'prevention':
                    return 'prevention_info', knowledge_base.get('prevention', {})
                elif category == 'treatment':
                    return 'treatment_info', knowledge_base.get('treatment', {})
                elif category == 'suggestions':
                    return 'suggestions_info', knowledge_base.get('suggestions', {})
    
    # If no keyword match, return general greeting
    return 'general', None

def generate_smart_suggestions(user_message: str, is_authenticated: bool) -> list:
    """Generate contextual suggestions based on user message and authentication status"""
    user_msg_lower = user_message.lower()
    
    # Base suggestions for authenticated users
    auth_suggestions = [
        "📊 View my analysis results",
        "📅 Book an appointment", 
        "🔍 Upload new retinal image",
        "📋 Download my reports"
    ]
    
    # Base suggestions for non-authenticated users
    guest_suggestions = [
        "🔬 Learn about AI diagnosis",
        "📝 Create an account",
        "👩‍⚕️ Find a specialist",
        "📚 Prevention guidelines"
    ]
    
    # Context-specific suggestions based on keywords
    if any(word in user_msg_lower for word in ['stage', 'severe', 'mild', 'moderate', 'proliferative']):
        return [
            "📊 Explain DR classification system",
            "🔍 What does my stage mean?",
            "💊 Treatment options for my stage",
            "⏰ How often should I get checked?"
        ]
    
    elif any(word in user_msg_lower for word in ['prevention', 'prevent', 'avoid', 'lifestyle']):
        return [
            "🥗 Diabetes diet recommendations",
            "🏃‍♂️ Exercise guidelines for diabetics",
            "🩺 Screening schedule",
            "📊 Monitor blood sugar levels"
        ]
    
    elif any(word in user_msg_lower for word in ['treatment', 'surgery', 'laser', 'injection']):
        return [
            "💉 Anti-VEGF injection therapy",
            "⚡ Laser photocoagulation",
            "🏥 When is surgery needed?",
            "📅 Schedule consultation"
        ]
    
    elif any(word in user_msg_lower for word in ['symptoms', 'vision', 'blurry', 'floaters']):
        return [
            "👁️ Recognize warning signs",
            "🚨 When to seek immediate care",
            "📞 Contact eye specialist",
            "🔍 Check my vision regularly"
        ]
    
    elif any(word in user_msg_lower for word in ['accuracy', 'ai', 'model', 'how does']):
        return [
            "🤖 How AI diagnosis works",
            "📈 Accuracy and validation",
            "🔬 Grad-CAM explanations",
            "⚡ Try the AI analysis"
        ]
    
    # Return appropriate base suggestions
    return auth_suggestions if is_authenticated else guest_suggestions

def generate_gemini_response(user_message: str, is_authenticated: bool = False) -> Dict:
    """Generate response using Google Gemini AI with medical context"""
    try:
        # Create specialized medical prompt with RetinalAI context
        system_prompt = """
You are an advanced AI medical assistant specializing in diabetic retinopathy and eye health, integrated into the RetinalAI diagnostic platform. Your knowledge is focused on:

**About RetinalAI Platform:**
- Advanced AI-powered diagnostic system using ensemble deep learning models (ResNet50, EfficientNet, VGG16, MobileNetV4)
- 95%+ diagnostic accuracy for diabetic retinopathy detection
- Real-time retinal image analysis with Grad-CAM explainability
- Clinical-grade reliability and FDA-compliance considerations
- Multi-stage classification system (0-4: No DR, Mild, Moderate, Severe, Proliferative DR)

**Your Expertise:**
- Diabetic retinopathy pathophysiology, stages, and progression
- Evidence-based prevention strategies and lifestyle modifications
- Current treatment guidelines and surgical interventions
- Risk factor assessment and patient education
- Screening recommendations and follow-up protocols
- Integration of AI diagnostics with clinical care

**Response Guidelines:**
- Provide accurate, evidence-based medical information
- Always include medical disclaimers for diagnosis/treatment advice
- Emphasize the importance of professional medical consultation
- Be empathetic and supportive while maintaining clinical accuracy
- Use clear, patient-friendly language with medical terminology when appropriate
- Include practical, actionable advice when relevant
- Reference current medical guidelines and research when applicable

**Important:** Always remind users that AI diagnostic tools support but do not replace professional medical judgment, and that any concerning symptoms should be evaluated by qualified healthcare providers.
        """
        
        # Enhanced prompt based on authentication status
        if is_authenticated:
            context_prompt = f"""
{system_prompt}

**Current User Context:** This user is logged into RetinalAI and has access to:
- AI-powered retinal image analysis
- Personal diagnostic history and reports
- Appointment booking with specialists
- Detailed explanation of AI predictions with Grad-CAM visualizations
- Progress tracking and trend analysis

Provide personalized guidance that leverages these platform features.
            """
        else:
            context_prompt = f"""
{system_prompt}

**Current User Context:** This is a visitor to RetinalAI. Encourage them to:
- Learn about diabetic retinopathy and prevention
- Understand the benefits of AI-assisted diagnosis
- Consider signing up for personalized analysis
- Seek professional medical evaluation when appropriate
            """
        
        # Construct the full prompt
        full_prompt = f"""
{context_prompt}

**User Question:** {user_message}

**Instructions:** Provide a comprehensive, medically accurate response that is helpful, empathetic, and actionable. Include relevant suggestions for follow-up questions or actions the user might want to take.
        """
        
        # Initialize Gemini model
        if not genai:
            raise Exception("Google Generative AI not available")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        # Generate relevant suggestions based on the user's question
        suggestions = generate_smart_suggestions(user_message, is_authenticated)
        
        return {
            'success': True,
            'response': response.text,
            'suggestions': suggestions,
            'source': 'gemini_ai'
        }
        
    except Exception as e:
        logger.error(f"Gemini AI error: {e}")
        # Fallback to basic response
        return {
            'success': False,
            'response': "I'm sorry, I'm experiencing technical difficulties right now. Please try again in a moment, or contact our support team if the issue persists.",
            'suggestions': ["Try again", "Contact support", "Visit FAQ"],
            'error': str(e)
        }

def generate_intelligent_chatbot_response(user_message: str, is_authenticated: bool = False) -> Dict:
    """Generate intelligent chatbot response based on user message"""
    try:
        category, info = find_best_match(user_message, KNOWLEDGE_BASE)
        
        # Handle different types of responses
        if category == 'product_info' and info:
            response = f"**About RetinalAI:**\n\n{info.get('description', '')}"
            formatted_content = {
                "title": "🔬 RetinalAI Advanced Diagnosis System",
                "sections": [
                    {
                        "heading": "🎯 Key Features",
                        "icon": "🤖",
                        "content": info.get('features', []),
                        "type": "list"
                    },
                    {
                        "heading": "📊 Performance Metrics",
                        "icon": "📈", 
                        "content": [
                            f"Accuracy: {info.get('accuracy', '')}",
                            f"Models: {info.get('models', '')}",
                            "Real-time analysis with Grad-CAM explanations"
                        ],
                        "type": "highlight"
                    }
                ],
                "highlights": [
                    "🎯 95%+ diagnostic accuracy",
                    "⚡ Real-time image analysis", 
                    "🔍 Transparent AI explanations",
                    "🏥 Clinical-grade reliability"
                ],
                "callToAction": "Ready to try our AI diagnosis system?",
                "severity": "info"
            }
            suggestions = ["🔍 How accurate is the diagnosis?", "🏥 Clinical validation studies", "⚡ Upload image for analysis"]
            
        elif category == 'disease_info' and info:
            response = f"**Diabetic Retinopathy:**\n\n{info.get('definition', '')}"
            formatted_content = {
                "title": "👁️ Understanding Diabetic Retinopathy",
                "sections": [
                    {
                        "heading": "🔍 Main Causes",
                        "icon": "⚠️",
                        "content": info.get('causes', []),
                        "type": "numbered"
                    },
                    {
                        "heading": "🚨 Symptoms to Watch",
                        "icon": "👁️",
                        "content": [
                            "Early Stage: " + ", ".join(info.get('symptoms', {}).get('early', [])),
                            "Advanced Stage: " + ", ".join(info.get('symptoms', {}).get('advanced', []))
                        ],
                        "type": "text"
                    },
                    {
                        "heading": "⚠️ Risk Factors",
                        "icon": "📋",
                        "content": info.get('risk_factors', []),
                        "type": "list"
                    }
                ],
                "highlights": [
                    "🚨 Often no early symptoms",
                    "📈 Risk increases with diabetes duration",
                    "🩺 Regular screening is crucial",
                    "💊 Blood sugar control is key"
                ],
                "callToAction": "Need personalized risk assessment?",
                "severity": "warning"
            }
            suggestions = ["📊 What are the stages?", "🛡️ How can I prevent it?", "🔍 Check my risk factors"]
            
        elif category == 'stages_info' and info:
            response = "**Diabetic Retinopathy Stages:**\n\nDetailed classification system"
            sections = []
            for stage_key, stage_info in info.items():
                sections.append({
                    "heading": f"Stage {stage_info.get('stage', '')}: {stage_info.get('name', '')}",
                    "icon": "📊",
                    "content": [
                        f"Description: {stage_info.get('description', '')}",
                        f"Key Features: {', '.join(stage_info.get('characteristics', []))}",
                        f"Recommended Action: {stage_info.get('action', '')}"
                    ],
                    "type": "text"
                })
            
            formatted_content = {
                "title": "📊 Complete Disease Classification",
                "sections": sections,
                "highlights": [
                    "🎯 5 distinct stages (0-4)",
                    "📈 Progressive severity levels",
                    "🏥 Specific treatment for each stage",
                    "⏰ Early detection saves vision"
                ],
                "callToAction": "Want to assess your current stage?",
                "severity": "info"
            }
            suggestions = ["🔍 What is my risk level?", "💊 How is it treated?", "🛡️ Prevention tips"]
            
        elif category == 'prevention_info' and info:
            response = "**Prevention & Screening Guidelines:**\n\nComprehensive prevention strategy"
            formatted_content = {
                "title": "🛡️ Complete Prevention Strategy",
                "sections": [
                    {
                        "heading": "🎯 Primary Prevention",
                        "icon": "🛡️",
                        "content": info.get('primary_prevention', []),
                        "type": "numbered"
                    },
                    {
                        "heading": "🔍 Screening Guidelines", 
                        "icon": "📋",
                        "content": info.get('screening', []),
                        "type": "list"
                    },
                    {
                        "heading": "💪 Lifestyle Modifications",
                        "icon": "🏃",
                        "content": info.get('lifestyle_modifications', []),
                        "type": "list"
                    }
                ],
                "highlights": [
                    "🎯 Blood sugar control (HbA1c < 7%)",
                    "💖 Blood pressure management", 
                    "👁️ Annual eye exams",
                    "🚭 Quit smoking"
                ],
                "callToAction": "Ready to start your prevention plan?",
                "severity": "success"
            }
            suggestions = ["💊 Treatment options", "🚨 What if I have symptoms?", "📊 Disease stages"]
            
        elif category == 'treatment_info' and info:
            response = "**Treatment Options:**\n\nComprehensive treatment approaches"
            formatted_content = {
                "title": "💊 Advanced Treatment Options",
                "sections": [
                    {
                        "heading": "💉 Medical Management",
                        "icon": "💊",
                        "content": info.get('medical_management', []),
                        "type": "numbered"
                    },
                    {
                        "heading": "🏥 Surgical Treatments",
                        "icon": "⚕️",
                        "content": info.get('surgical_treatments', []),
                        "type": "list"
                    },
                    {
                        "heading": "📋 Follow-up Care",
                        "icon": "📅",
                        "content": info.get('follow_up', []),
                        "type": "list"
                    }
                ],
                "highlights": [
                    "⚡ Early treatment prevents vision loss",
                    "💉 Anti-VEGF injections highly effective",
                    "🔬 Laser therapy for advanced stages",
                    "👨‍⚕️ Multidisciplinary care approach"
                ],
                "callToAction": "Need help finding a specialist?",
                "severity": "info"
            }
            suggestions = ["🛡️ Prevention tips", "🔍 How to find a specialist?", "📊 What to expect?"]
            
        elif category == 'suggestions_info' and info:
            response = "**Personalized Recommendations:**\n\nTailored guidance based on your condition"
            immediate_sections = []
            immediate_actions = info.get('immediate_actions', {})
            for condition, actions in immediate_actions.items():
                condition_name = condition.replace('_', ' ').title()
                immediate_sections.append({
                    "heading": f"For {condition_name}",
                    "icon": "🎯",
                    "content": actions,
                    "type": "numbered"
                })
            
            formatted_content = {
                "title": "🎯 Personalized Action Plan",
                "sections": immediate_sections + [{
                    "heading": "🔄 Long-term Care Strategy",
                    "icon": "🏥",
                    "content": info.get('long_term_care', []),
                    "type": "list"
                }],
                "highlights": [
                    "📋 Personalized recommendations",
                    "👥 Healthcare team coordination",
                    "📱 Digital health tools",
                    "📚 Ongoing education"
                ],
                "callToAction": "Ready to implement your care plan?",
                "severity": "success"
            }
            suggestions = ["📅 Book an appointment", "🔍 Find a doctor", "📊 Learn about stages"]
            
        else:
            # General greeting or unmatched query
            greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
            if any(greeting in user_message.lower() for greeting in greetings):
                response = "Hello! I'm your RetinalAI Medical Assistant."
                formatted_content = {
                    "title": "🩺 Welcome to RetinalAI",
                    "sections": [
                        {
                            "heading": "🤖 What I Can Help With",
                            "icon": "💡",
                            "content": [
                                "Diabetic retinopathy information and stages",
                                "AI diagnosis system explanations",
                                "Prevention strategies and screening",
                                "Treatment options and recommendations",
                                "Risk assessment and management"
                            ],
                            "type": "list"
                        }
                    ],
                    "highlights": [
                        "🎯 Expert medical guidance",
                        "🔬 AI-powered insights",
                        "📊 Personalized recommendations",
                        "🏥 Evidence-based information"
                    ],
                    "callToAction": "What would you like to know first?",
                    "severity": "info"
                }
            else:
                response = "I'd be happy to provide detailed medical information."
                formatted_content = {
                    "title": "🔍 Ask Me Anything About",
                    "sections": [
                        {
                            "heading": "📚 Available Topics",
                            "icon": "📖",
                            "content": [
                                "Diabetic retinopathy (causes, symptoms, stages)",
                                "RetinalAI diagnosis system (accuracy, models)",
                                "Prevention strategies (diet, exercise, monitoring)",
                                "Treatment options (medical, surgical)",
                                "Risk factors and management"
                            ],
                            "type": "list"
                        }
                    ],
                    "callToAction": "Please ask a specific question for detailed guidance!",
                    "severity": "info"
                }
            
            suggestions = [
                "👁️ What is diabetic retinopathy?",
                "🔬 How does RetinalAI work?",
                "🛡️ Prevention tips",
                "📊 Stages of diabetic retinopathy",
                "💊 Treatment options"
            ]
        
        # Add appointment booking suggestion for authenticated users
        if is_authenticated and "📅 Book appointment" not in suggestions:
            suggestions.append("📅 Book appointment")
        
        return {
            'success': True,
            'response': response,
            'suggestions': suggestions,
            'formattedContent': formatted_content
        }
        
    except Exception as e:
        logger.error(f"Chatbot response generation error: {e}")
        return {
            'success': False,
            'response': "I'm sorry, I'm having trouble processing your request right now. Please try again or contact support if the issue persists.",
            'suggestions': ["🔄 Try again", "📞 Contact support"]
        }

def generate_pdf_report(report_data: Dict, output_path: str):
    """Generate PDF report"""
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Diabetic Retinopathy Analysis Report")
        c.setFont("Helvetica", 12)
        y_position = height - 100
        c.drawString(50, y_position, f"Patient: {report_data.get('patient_name', 'N/A')}")
        y_position -= 20
        c.drawString(50, y_position, f"Date: {report_data.get('date', 'N/A')}")
        y_position -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Analysis Results:")
        y_position -= 30
        c.setFont("Helvetica", 12)
        final_prediction_text = DR_STAGES.get(report_data['final_prediction'], 'Unknown')
        c.drawString(50, y_position, f"Final Prediction: {final_prediction_text}")
        y_position -= 20
        c.drawString(50, y_position, f"Confidence: {report_data['confidence']:.2%}")
        y_position -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Individual Model Predictions:")
        y_position -= 20
        c.setFont("Helvetica", 10)
        for model_name, pred in report_data.get('predictions', {}).items():
            if 'error' not in pred:
                pred_text = DR_STAGES.get(pred['predicted_class'], 'Unknown')
                c.drawString(70, y_position, f"{model_name}: {pred_text} ({pred['confidence']})")
            else:
                c.drawString(70, y_position, f"{model_name}: Error ({pred['error']})")
            y_position -= 15
        # Add original image
        if report_data.get('original_image_path'):
            y_position -= 20
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, "Original Fundus Image:")
            y_position -= 20
            try:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], report_data['original_image_path'])
                if os.path.exists(img_path):
                    # Resize image to fit within PDF
                    img = Image.open(img_path)
                    aspect_ratio = img.width / img.height
                    max_width = width / 2 - 50 # Half width of page minus margins
                    max_height = 200 # Max height for image
                    if img.width > max_width or img.height > max_height:
                        if aspect_ratio > 1: # Wider than tall
                            img_display_width = max_width
                            img_display_height = max_width / aspect_ratio
                        else: # Taller than wide
                            img_display_height = max_height
                            img_display_width = max_height * aspect_ratio
                    else:
                        img_display_width = img.width
                        img_display_height = img.height

                    c.drawImage(ImageReader(img_path), 50, y_position - img_display_height - 10,
                                width=img_display_width, height=img_display_height, preserveAspectRatio=True)
                    y_position -= (img_display_height + 30)
            except Exception as e:
                logger.error(f"Error adding original image to PDF: {e}")
        c.save()
        logger.info(f"PDF report generated at {output_path}")
    except Exception as e:
        logger.error(f"PDF report generation failed: {e}")
        raise

@app.route('/api/ai-insights', methods=['POST'])
@login_required
def ai_insights_api():
    """Generate AI insights for prediction results using Gemini AI"""
    try:
        data = request.get_json()
        prediction = data.get('prediction', '')
        confidence = data.get('confidence', '')
        user_context = data.get('user_context', {})
        
        if not prediction:
            return jsonify({
                'success': False,
                'message': 'Prediction result is required'
            }), 400
        
        # Create specialized prompt for AI insights
        insights_prompt = f"""
You are an expert ophthalmologist and AI specialist providing detailed insights about a diabetic retinopathy diagnosis from RetinalAI system.

**DIAGNOSIS DETAILS:**
- Prediction: {prediction}
- Confidence: {confidence}
- Patient: Authenticated user seeking understanding

**YOUR TASK:**
Provide a comprehensive, medically accurate analysis covering:

1. **WHY THIS PREDICTION** (2-3 sentences):
   - Explain what clinical features or patterns the AI likely detected
   - What specific retinal changes are associated with this diagnosis
   - How the AI models work together to reach this conclusion

2. **PREVENTION STRATEGIES** (4-5 actionable points):
   - Evidence-based lifestyle modifications
   - Blood glucose management strategies
   - Blood pressure and cholesterol control
   - Regular monitoring schedules
   - Dietary recommendations

3. **PRECAUTIONS & WARNING SIGNS** (3-4 critical points):
   - Symptoms that require immediate medical attention
   - When to schedule urgent eye exams
   - Activity restrictions if any
   - Medication compliance importance

4. **PERSONALIZED SUGGESTIONS** (4-5 specific recommendations):
   - Next steps for this specific diagnosis stage
   - Screening frequency recommendations
   - Specialist referrals if needed
   - Technology tools for monitoring
   - Support resources and education

**RESPONSE GUIDELINES:**
- Use clear, patient-friendly language
- Be encouraging and supportive while being medically accurate
- Include specific, actionable advice
- Emphasize the importance of professional medical care
- Reference current medical guidelines
- Always include appropriate medical disclaimers

**MEDICAL DISCLAIMER:**
Always conclude with: "This AI analysis is for educational purposes only and does not replace professional medical advice. Please consult with your healthcare provider or ophthalmologist for personalized diagnosis and treatment recommendations."
        """
        
        try:
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Generate insights
            response = model.generate_content(insights_prompt)
            
            # Clean up the response text
            cleaned_insights = clean_markdown_text(response.text)
            
            return jsonify({
                'success': True,
                'insights': cleaned_insights,
                'source': 'gemini_ai_insights'
            })
            
        except Exception as e:
            logger.error(f"Gemini AI insights error: {e}")
            # Fallback response
            fallback_insights = generate_fallback_insights(prediction, confidence)
            return jsonify({
                'success': True,
                'insights': fallback_insights,
                'source': 'fallback_insights'
            })
        
    except Exception as e:
        logger.error(f"AI insights API error: {e}")
        return jsonify({
            'success': False,
            'message': 'Sorry, I encountered an error generating insights. Please try again.',
            'error': str(e)
        }), 500

def generate_fallback_insights(prediction: str, confidence: str) -> str:
    """Generate fallback insights when Gemini is unavailable"""
    
    # Determine severity level
    if 'No DR' in prediction:
        severity = 'no_dr'
    elif 'Mild' in prediction:
        severity = 'mild'
    elif 'Moderate' in prediction:
        severity = 'moderate'
    elif 'Severe' in prediction:
        severity = 'severe'
    elif 'Proliferative' in prediction:
        severity = 'proliferative'
    else:
        severity = 'unknown'
    
    insights_templates = {
        'no_dr': {
            'why': "The AI detected no visible signs of diabetic retinopathy in your retinal image. All four neural networks analyzed your retina and found healthy blood vessels, no microaneurysms, hemorrhages, or other abnormalities typically associated with diabetic eye disease.",
            'prevention': [
                "Maintain excellent blood glucose control (HbA1c < 7%)",
                "Monitor blood pressure regularly (target < 140/90 mmHg)",
                "Follow a diabetes-friendly diet rich in vegetables and whole grains",
                "Exercise regularly (150 minutes of moderate activity per week)",
                "Take prescribed diabetes medications consistently"
            ],
            'precautions': [
                "Schedule annual comprehensive eye exams",
                "Report any sudden vision changes immediately",
                "Monitor blood sugar levels as recommended by your doctor",
                "Maintain healthy cholesterol levels"
            ],
            'suggestions': [
                "Continue current diabetes management strategies",
                "Consider diabetes education classes for optimization",
                "Use glucose monitoring apps for better tracking",
                "Maintain regular follow-up with your endocrinologist",
                "Keep a log of blood pressure and glucose readings"
            ]
        },
        'mild': {
            'why': "The AI identified early signs of diabetic retinopathy, specifically microaneurysms - small balloon-like swellings in retinal blood vessels. This indicates that diabetes has begun to affect your eye's blood vessels, but the damage is still minimal and manageable.",
            'prevention': [
                "Optimize blood glucose control immediately (target HbA1c < 7%)",
                "Intensify blood pressure management (target < 130/80 mmHg)",
                "Implement strict cholesterol control measures",
                "Quit smoking if applicable - it accelerates retinal damage",
                "Consider anti-inflammatory diet modifications"
            ],
            'precautions': [
                "Schedule eye exams every 6-12 months instead of annually",
                "Watch for floaters, blurred vision, or vision loss",
                "Report any sudden vision changes to your eye doctor immediately",
                "Ensure consistent medication compliance"
            ],
            'suggestions': [
                "Work with endocrinologist to optimize diabetes management",
                "Consider referral to diabetic educator",
                "Use continuous glucose monitoring if recommended",
                "Schedule follow-up retinal photography in 6 months",
                "Join diabetic retinopathy support groups"
            ]
        },
        'moderate': {
            'why': "The AI detected moderate diabetic retinopathy with multiple retinal abnormalities including microaneurysms, small hemorrhages, and possibly hard exudates or cotton wool spots. This indicates progressive damage to retinal blood vessels requiring closer monitoring.",
            'prevention': [
                "Achieve strict glycemic control (HbA1c < 6.5% if safely achievable)",
                "Maintain optimal blood pressure (< 130/80 mmHg)",
                "Consider cardioprotective medications as prescribed",
                "Implement intensive lifestyle interventions",
                "Regular monitoring of kidney function"
            ],
            'precautions': [
                "Schedule eye exams every 3-6 months",
                "Immediately report any vision changes, floaters, or flashing lights",
                "Monitor for symptoms of macular edema (central vision changes)",
                "Avoid activities that cause sudden blood pressure spikes"
            ],
            'suggestions': [
                "Consider referral to retinal specialist",
                "Discuss anti-VEGF treatment options with ophthalmologist",
                "Optimize diabetes team care coordination",
                "Consider OCT imaging for macular edema screening",
                "Explore diabetes technology solutions (CGM, insulin pumps)"
            ]
        },
        'severe': {
            'why': "The AI identified severe nonproliferative diabetic retinopathy with extensive retinal damage including numerous hemorrhages, microaneurysms, venous abnormalities, and intraretinal microvascular abnormalities. This stage requires urgent ophthalmologic intervention.",
            'prevention': [
                "Emergency optimization of diabetes management",
                "Immediate blood pressure control (< 130/80 mmHg)",
                "Aggressive cardiovascular risk reduction",
                "Consider hospitalization for diabetes management if needed",
                "Implement comprehensive diabetes care team approach"
            ],
            'precautions': [
                "Schedule urgent ophthalmologic consultation (within 1-2 weeks)",
                "Monitor for proliferative changes every 2-3 months",
                "Immediately report any new floaters, flashing lights, or vision loss",
                "Avoid activities that increase intraocular pressure"
            ],
            'suggestions': [
                "Immediate referral to retinal specialist required",
                "Consider panretinal photocoagulation evaluation",
                "Optimize multidisciplinary diabetes care",
                "Regular OCT and fluorescein angiography monitoring",
                "Prepare for potential laser treatment or injections"
            ]
        },
        'proliferative': {
            'why': "The AI detected proliferative diabetic retinopathy with neovascularization - abnormal new blood vessel growth on the retina or optic disc. This advanced stage poses significant risk for severe vision loss and requires immediate treatment.",
            'prevention': [
                "Immediate intensive diabetes management in hospital setting if needed",
                "Emergency blood pressure optimization",
                "Comprehensive cardiovascular risk assessment and management",
                "Immediate ophthalmologic intervention planning",
                "Consider systemic anti-VEGF therapy consultation"
            ],
            'precautions': [
                "Seek immediate ophthalmologic emergency care",
                "Monitor for vitreous hemorrhage or retinal detachment daily",
                "Avoid straining, heavy lifting, or activities that increase eye pressure",
                "Report any sudden vision loss or curtain-like vision changes immediately"
            ],
            'suggestions': [
                "Emergency retinal specialist consultation (within 24-48 hours)",
                "Immediate panretinal photocoagulation evaluation",
                "Consider anti-VEGF injection therapy",
                "Prepare for possible vitrectomy surgery",
                "Arrange intensive diabetes management hospitalization if needed"
            ]
        }
    }
    
    template = insights_templates.get(severity, insights_templates['unknown'] if 'unknown' in insights_templates else insights_templates['no_dr'])
    
    insights = f"""**WHY THIS PREDICTION:**
{template['why']}

**PREVENTION STRATEGIES:**
{chr(10).join(f"• {item}" for item in template['prevention'])}

**PRECAUTIONS & WARNING SIGNS:**
{chr(10).join(f"• {item}" for item in template['precautions'])}

**PERSONALIZED SUGGESTIONS:**
{chr(10).join(f"• {item}" for item in template['suggestions'])}

**MEDICAL DISCLAIMER:**
This AI analysis is for educational purposes only and does not replace professional medical advice. Please consult with your healthcare provider or ophthalmologist for personalized diagnosis and treatment recommendations."""
    
    return insights

def clean_markdown_text(text: str) -> str:
    """Clean markdown symbols from text for better display"""
    import re
    
    # Remove markdown headers (###, ##, etc.)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Remove excessive asterisks but keep content
    text = re.sub(r'\*{3,}([^*]+)\*{3,}', r'\1', text)  # Remove triple+ asterisks
    text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)     # Remove double asterisks
    
    # Clean up extra whitespace and line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple line breaks
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading whitespace
    
    return text.strip()

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """Chatbot API endpoint using Google Gemini AI for intelligent responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        is_authenticated = data.get('isAuthenticated', False)
        
        if not user_message:
            return jsonify({
                'success': False,
                'message': 'Message is required'
            }), 400
        
        # Handle special commands for authenticated users
        if is_authenticated:
            if 'book appointment' in user_message.lower() or 'schedule appointment' in user_message.lower():
                return jsonify({
                    'success': True,
                    'response': "I can help you book an appointment! Please use the appointment booking feature in your dashboard to schedule a consultation with one of our specialists. You can choose from available doctors and time slots that work best for you.",
                    'suggestions': ["View my results", "Find a doctor", "Prevention tips"],
                    'source': 'system_command'
                })
            
            if 'my results' in user_message.lower() or 'my reports' in user_message.lower():
                return jsonify({
                    'success': True,
                    'response': "You can view all your diagnosis results and reports in the Results section of your dashboard. Each report includes detailed analysis, AI predictions, and doctor notes if available.",
                    'suggestions': ["Book an appointment", "Download report", "Share with doctor"],
                    'source': 'system_command'
                })
        
        # Generate response using Gemini AI
        result = generate_gemini_response(user_message, is_authenticated)
        
        # Clean up the response text if it exists
        if result.get('success') and 'response' in result:
            result['response'] = clean_markdown_text(result['response'])
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chatbot API error: {e}")
        return jsonify({
            'success': False,
            'message': 'Sorry, I encountered an error. Please try again.',
            'error': str(e)
        }), 500

# Root path for testing
@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the RetinalAI API!'})

@app.route('/api/register', methods=['POST'])
def register_api():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')
    full_name = data.get('full_name', data.get('username', ''))  # Use full_name if provided, fallback to username
    if not all([username, email, password, role, full_name]):
        return jsonify({'success': False, 'message': 'All fields are required!'}), 400

    password_hash = generate_password_hash(password)
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        c.execute('INSERT INTO users (username, email, password_hash, role, full_name) VALUES (?, ?, ?, ?, ?)',
                 (username, email, password_hash, role, full_name))
        conn.commit()
        conn.close()
        logger.info(f"User {username} registered")
        return jsonify({'success': True, 'message': 'Registration successful! Please login.'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Username or email already exists!'}), 409
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return jsonify({'success': False, 'message': 'Registration failed. Please try again.'}), 500

@app.route('/api/login', methods=['POST'])
def login_api():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    # Removed req_role requirement - login with any valid user regardless of role sent from frontend

    if not all([email, password]):
        return jsonify({'success': False, 'message': 'Email and password are required!'}), 400

    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        c.execute('SELECT id, username, email, password_hash, role, full_name FROM users WHERE email = ?', (email,))
        user_db = c.fetchone()
        conn.close()

        if user_db and check_password_hash(user_db[3], password):
            # No role check - use the user's actual role from DB
            session['user_id'] = user_db[0]
            session['username'] = user_db[1]
            session['email'] = user_db[2]
            session['role'] = user_db[4]
            session['full_name'] = user_db[5]
            logger.info(f"User {user_db[1]} logged in with role {user_db[4]}")
            return jsonify({
                'success': True,
                'message': 'Login successful!',
                'user': {
                    'id': user_db[0],
                    'username': user_db[1],
                    'email': user_db[2],
                    'role': user_db[4]
                }
            }), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password!'}), 401
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout_api():
    username = session.get('username', 'Unknown')
    session.clear()
    logger.info(f"User {username} logged out")
    return jsonify({'success': True, 'message': 'You have been logged out.'}), 200

@app.route('/api/user', methods=['GET'])
@login_required
def get_user_data():
    """Endpoint to get current logged-in user's data"""
    return jsonify({
        'success': True,
        'user': {
            'id': session['user_id'],
            'username': session['username'],
            'email': session['email'],
            'role': session['role'],
            'full_name': session['full_name']
        }
    }), 200

@app.route('/api/predict-clinical', methods=['POST'])
@role_required('patient')
def predict_clinical_api():
    """Predict diabetic retinopathy using clinical data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No clinical data provided'}), 400
        
        logger.info(f"Received clinical data: {data}")
        
        # Validate required fields
        required_fields = ['Age', 'HbA1c', 'Gender', 'Stage', 'Blood_Pressure']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'success': False, 
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate data ranges
        try:
            age = float(data.get('Age', 0))
            hba1c = float(data.get('HbA1c', 0))
            
            if not (18 <= age <= 120):
                return jsonify({
                    'success': False, 
                    'message': 'Age must be between 18 and 120 years'
                }), 400
                
            if not (4.0 <= hba1c <= 15.0):
                return jsonify({
                    'success': False, 
                    'message': 'HbA1c must be between 4.0% and 15.0%'
                }), 400
                
        except (ValueError, TypeError):
            return jsonify({
                'success': False, 
                'message': 'Invalid numeric values for Age or HbA1c'
            }), 400
        
        # Check if clinical model is available
        if clinical_model is None:
            return jsonify({
                'success': False, 
                'message': 'Clinical prediction model is not available. Please contact support.'
            }), 503
        
        # Make prediction
        prediction_result = predict_with_clinical_model(data)
        
        if 'error' in prediction_result:
            logger.error(f"Prediction error: {prediction_result['error']}")
            return jsonify({
                'success': False, 
                'message': f"Prediction failed: {prediction_result['error']}"
            }), 500
        
        # Save to database (similar to image predictions)
        report_id = None
        try:
            conn = sqlite3.connect('diabetic_retinopathy.db')
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            c = conn.cursor()
            
            # Store clinical data and prediction
            clinical_data_json = json.dumps(data)
            prediction_json = json.dumps(prediction_result)
            
            c.execute('''INSERT INTO reports (user_id, filename, predictions, final_prediction, confidence)
                        VALUES (?, ?, ?, ?, ?)''',
                     (session['user_id'], 'clinical_data', prediction_json, 
                      prediction_result['predicted_class'], 
                      float(prediction_result['confidence'].rstrip('%')) / 100))
            
            report_id = c.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Clinical prediction saved for user {session['user_id']}: {prediction_result['prediction_text']}")
            
        except Exception as db_error:
            logger.error(f"Database error saving clinical prediction: {db_error}")
            # Continue without saving to DB
        
        # Return prediction result
        return jsonify({
            'success': True,
            'message': 'Clinical analysis completed successfully.',
            'result': {
                'prediction': prediction_result['prediction_text'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'risk_level': 'High' if prediction_result['predicted_class'] == 1 else 'Low',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_id': report_id
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Clinical prediction error: {e}")
        return jsonify({
            'success': False, 
            'message': f'Error processing clinical data: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
@role_required('patient')
def predict_api():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '' or file.filename is None:
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    if not allowed_file(file.filename or ''):
        return jsonify({'success': False, 'message': 'Invalid file type. Only PNG, JPG, JPEG are allowed.'}), 400

    filename = secure_filename(file.filename or 'unknown.jpg')
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    try:
        file.save(filepath)

        # Stage 1: Check if the image is a fundus image using loaded model or fallback
        is_fundus, confidence, error = predict_stage1_fundus(filepath)
        
        if error:
            logger.error(f"Stage 1 prediction error: {error}")
            # Continue with processing but log the error
            
        if not is_fundus:
            # If not a fundus image, return appropriate message - NO Stage 2 processing
            return jsonify({
                'success': False,
                'message': 'The uploaded image does not appear to be a retinal fundus image. Please upload a proper fundus photograph for analysis.',
                'is_fundus': False,
                'confidence': f"{confidence:.2%}" if isinstance(confidence, (int, float)) else confidence
            }), 400

        # Stage 2: Proceed with DR classification ONLY if Stage 1 confirms fundus image
        predictions = predict_with_models(filepath)
        final_prediction_idx, ensemble_confidence_val, _ = majority_voting(predictions)
        final_prediction_text = DR_STAGES[final_prediction_idx]

        # Generate Grad-CAM for all models
        gradcam_all = generate_all_gradcam_explanations(filepath, final_prediction_idx)

        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        c.execute('''INSERT INTO reports (user_id, filename, predictions, final_prediction, confidence)
                    VALUES (?, ?, ?, ?, ?)''',
                 (session['user_id'], unique_filename, json.dumps(predictions), final_prediction_idx, ensemble_confidence_val))
        report_id = c.lastrowid
        conn.commit()
        conn.close()

        # Prepare individual model predictions for frontend
        formatted_predictions = {}
        for arch, pred_data in predictions.items():
            if 'error' not in pred_data:
                formatted_predictions[arch] = {
                    'prediction': pred_data['prediction_text'],
                    'confidence': pred_data['confidence']
                }
            else:
                formatted_predictions[arch] = {'error': pred_data['error']}

        return jsonify({
            'success': True,
            'message': 'Analysis completed successfully.',
            'report': {
                'id': report_id,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'final_result': final_prediction_text,
                'confidence': f"{ensemble_confidence_val:.2%}",
                'image': f"/uploads/{unique_filename}", # URL to the uploaded image
                'models': formatted_predictions,
                'explanations': {
                    'gradcam': gradcam_all
                },
                'risk_factors': ["High blood sugar", "High blood pressure"] if final_prediction_idx > 0 else [], # Example
                'recommendations': ["Consult an ophthalmologist", "Monitor blood sugar regularly"] if final_prediction_idx > 0 else ["Continue regular eye check-ups", "Maintain healthy lifestyle"], # Example
                'is_fundus': True,
                'fundus_confidence': f"{confidence:.2%}" if isinstance(confidence, (int, float)) else confidence
            }
        }), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500

@app.route('/api/results', methods=['GET'])
@role_required('patient')
def get_patient_results_api():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        # Fetch reports for the logged-in patient
        c.execute('''SELECT id, filename, predictions, final_prediction, confidence, created_at, doctor_notes
                    FROM reports WHERE user_id = ? ORDER BY created_at DESC''', (session['user_id'],))
        reports_db = c.fetchall()
        conn.close()

        results = []
        for r in reports_db:
            try:
                predictions_data = json.loads(r[2])
                
                # Handle different types of predictions
                if r[1] == 'clinical_data':
                    # Clinical prediction - extract result from predictions JSON
                    if 'prediction_text' in predictions_data:
                        result_text = predictions_data['prediction_text']
                    else:
                        result_text = 'Clinical Analysis'
                    models_display = {'clinical': {'prediction': result_text, 'confidence': 'N/A'}}
                else:
                    # Image prediction - use DR_STAGES mapping
                    final_pred_idx = r[3]
                    if isinstance(final_pred_idx, int) and final_pred_idx in DR_STAGES:
                        result_text = DR_STAGES[final_pred_idx]
                    else:
                        result_text = 'Unknown Result'
                    
                    models_display = {}
                    for arch, pred_detail in predictions_data.items():
                        if isinstance(pred_detail, dict) and 'error' not in pred_detail:
                            models_display[arch] = {
                                'prediction': pred_detail.get('prediction_text', 'Unknown'),
                                'confidence': pred_detail.get('confidence', 'N/A')
                            }
                        elif isinstance(pred_detail, dict):
                            models_display[arch] = {'error': pred_detail.get('error', 'Unknown error')}

                # Format confidence
                confidence_val = r[4]
                if isinstance(confidence_val, (int, float)):
                    confidence_str = f"{confidence_val:.2%}"
                else:
                    confidence_str = str(confidence_val)

                results.append({
                    'id': r[0],
                    'filename': r[1],
                    'result': result_text,
                    'confidence': confidence_str,
                    'date': datetime.strptime(r[5], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),
                    'models': models_display,
                    'image': f"/uploads/{r[1]}" if r[1] != 'clinical_data' else None,
                    'doctor_notes': r[6] or ''
                })
            except Exception as row_error:
                logger.error(f"Error processing row {r[0]}: {row_error}")
                # Add a fallback entry for corrupted data
                results.append({
                    'id': r[0],
                    'filename': r[1],
                    'result': 'Data Error',
                    'confidence': 'N/A',
                    'date': r[5][:10] if r[5] else 'Unknown',
                    'models': {},
                    'image': None,
                    'doctor_notes': 'Error loading this report'
                })
                
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error fetching patient results: {e}")
        return jsonify({'success': False, 'message': 'Error fetching results.'}), 500

@app.route('/api/result/<int:report_id>', methods=['GET'])
@login_required
def get_result_detail_api(report_id):
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        # Ensure user can only view their own reports or reports shared with them
        if session['role'] == 'patient':
            c.execute('''SELECT r.id, r.filename, r.predictions, r.final_prediction, r.confidence, r.created_at, r.doctor_notes
                        FROM reports r WHERE r.id = ? AND r.user_id = ?''', (report_id, session['user_id']))
        elif session['role'] == 'doctor':
             c.execute('''SELECT r.id, r.filename, r.predictions, r.final_prediction, r.confidence, r.created_at, r.doctor_notes, u.full_name, u.email
                        FROM reports r JOIN shared_reports sr ON r.id = sr.report_id JOIN users u ON r.user_id = u.id
                        WHERE r.id = ? AND sr.doctor_id = ?''', (report_id, session['user_id']))
        else:
            conn.close()
            return jsonify({'success': False, 'message': 'Unauthorized access to report.'}), 403

        report_db = c.fetchone()
        conn.close()

        if not report_db:
            return jsonify({'success': False, 'message': 'Report not found or unauthorized access.'}), 404

        try:
            predictions_data = json.loads(report_db[2])
            
            # Handle different types of predictions
            if report_db[1] == 'clinical_data':
                # Clinical prediction
                if 'prediction_text' in predictions_data:
                    final_result = predictions_data['prediction_text']
                else:
                    final_result = 'Clinical Analysis'
                models_display = {'clinical': {'prediction': final_result, 'confidence': 'N/A'}}
            else:
                # Image prediction
                final_pred_idx = report_db[3]
                if isinstance(final_pred_idx, int) and final_pred_idx in DR_STAGES:
                    final_result = DR_STAGES[final_pred_idx]
                else:
                    final_result = 'Unknown Result'
                    
                models_display = {}
                for arch, pred_detail in predictions_data.items():
                    if isinstance(pred_detail, dict) and 'error' not in pred_detail:
                        models_display[arch] = {
                            'prediction': pred_detail.get('prediction_text', 'Unknown'),
                            'confidence': pred_detail.get('confidence', 'N/A')
                        }
                    elif isinstance(pred_detail, dict):
                        models_display[arch] = {'error': pred_detail.get('error', 'Unknown error')}
        except Exception as parse_error:
            logger.error(f"Error parsing predictions data: {parse_error}")
            final_result = 'Data Error'
            models_display = {}

        # Generate explanation images for all models (Grad-CAM) - only for image predictions
        gradcam_all = {}
        if report_db[1] != 'clinical_data':
            report_filepath = os.path.join(app.config['UPLOAD_FOLDER'], report_db[1])
            if os.path.exists(report_filepath):
                final_pred_idx = report_db[3]
                if isinstance(final_pred_idx, int):
                    gradcam_all = generate_all_gradcam_explanations(report_filepath, final_pred_idx)

        # Format confidence
        confidence_val = report_db[4]
        if isinstance(confidence_val, (int, float)):
            confidence_str = f"{confidence_val:.2%}"
        else:
            confidence_str = str(confidence_val)

        report_data = {
            'id': report_db[0],
            'filename': report_db[1],
            'final_result': final_result,
            'confidence': confidence_str,
            'date': datetime.strptime(report_db[5], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),
            'doctor_notes': report_db[6] or '',
            'image': f"/uploads/{report_db[1]}" if report_db[1] != 'clinical_data' else None,
            'models': models_display,
            'explanations': {
                'gradcam': gradcam_all
            },
            'risk_factors': ["High blood sugar", "High blood pressure"] if report_db[3] and isinstance(report_db[3], int) and report_db[3] > 0 else [],
            'recommendations': ["Consult an ophthalmologist", "Monitor blood sugar regularly"] if report_db[3] and isinstance(report_db[3], int) and report_db[3] > 0 else ["Continue regular eye check-ups", "Maintain healthy lifestyle"]
        }
        
        # Add patient information for doctors
        if session['role'] == 'doctor' and len(report_db) > 8:
            report_data['patient_name'] = report_db[7]  # full_name
            report_data['patient_email'] = report_db[8]  # email

        return jsonify(report_data), 200
    except Exception as e:
        logger.error(f"Error fetching result detail: {e}")
        return jsonify({'success': False, 'message': 'Error fetching report details.'}), 500


@app.route('/api/download-report/<int:report_id>', methods=['GET'])
@login_required
def download_report_api(report_id):
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        # Ensure user can only download their own reports or reports shared with them
        if session['role'] == 'patient':
            c.execute('''SELECT r.id, r.filename, r.predictions, r.final_prediction, r.confidence, r.created_at, u.full_name
                        FROM reports r JOIN users u ON r.user_id = u.id
                        WHERE r.id = ? AND r.user_id = ?''', (report_id, session['user_id']))
        elif session['role'] == 'doctor':
             c.execute('''SELECT r.id, r.filename, r.predictions, r.final_prediction, r.confidence, r.created_at, u.full_name
                        FROM reports r JOIN shared_reports sr ON r.id = sr.report_id JOIN users u ON r.user_id = u.id
                        WHERE r.id = ? AND sr.doctor_id = ?''', (report_id, session['user_id']))
        else:
            conn.close()
            return jsonify({'success': False, 'message': 'Unauthorized access to report.'}), 403

        report = c.fetchone()
        conn.close()

        if not report:
            return jsonify({'success': False, 'message': 'Report not found!'}), 404

        report_data = {
            'patient_name': report[6], # full_name of the patient
            'date': report[5],
            'final_prediction': report[3],
            'confidence': report[4],
            'predictions': json.loads(report[2]),
            'original_image_path': report[1] # Pass filename to PDF generator
        }
        pdf_filename = f"report_{report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join('static/reports', pdf_filename)
        generate_pdf_report(report_data, pdf_path)
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    except Exception as e:
        logger.error(f"Report download error: {e}")
        return jsonify({'success': False, 'message': 'Error generating report.'}), 500

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serve uploaded images securely (ensure logged in)"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/api/doctors', methods=['GET'])
@role_required('patient')
def get_doctors_api():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        c.execute('SELECT id, full_name, email FROM users WHERE role = "doctor"')
        doctors_db = c.fetchall()
        conn.close()
        doctors = [{'id': d[0], 'name': d[1], 'email': d[2]} for d in doctors_db]
        return jsonify(doctors), 200
    except Exception as e:
        logger.error(f"Error fetching doctors: {e}")
        return jsonify({'success': False, 'message': 'Error fetching doctors.'}), 500

@app.route('/api/doctor-appointments', methods=['GET'])
@role_required('doctor')
def doctor_appointments_api():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        c.execute('''SELECT a.id, a.appointment_date, a.appointment_time, a.reason, a.status, u.full_name as patient_name, u.email as patient_email
                    FROM appointments a
                    JOIN users u ON a.patient_id = u.id
                    WHERE a.doctor_id = ?
                    ORDER BY a.appointment_date DESC, a.appointment_time DESC''', (session['user_id'],))
        appointments_db = c.fetchall()
        conn.close()

        appointments = []
        for appt in appointments_db:
            appointments.append({
                'id': appt[0],
                'date': appt[1],
                'time': appt[2],
                'reason': appt[3],
                'status': appt[4],
                'patient_name': appt[5],
                'patient_email': appt[6]
            })
        return jsonify(appointments), 200
    except Exception as e:
        logger.error(f"Error fetching doctor appointments: {e}")
        return jsonify({'success': False, 'message': 'Error loading appointments.'}), 500

@app.route('/api/doctor-appointments/<int:appointment_id>/status', methods=['POST'])
@role_required('doctor')
def update_doctor_appointment_status_api(appointment_id: int):
    try:
        data = request.get_json()
        new_status = data.get('status')

        # Allow only specific statuses doctors can set
        if new_status not in ['accepted', 'declined', 'pending']:
            return jsonify({'success': False, 'message': 'Invalid status provided.'}), 400

        conn = sqlite3.connect('diabetic_retinopathy.db')
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        # Ensure this appointment belongs to the logged-in doctor
        c.execute('SELECT id FROM appointments WHERE id = ? AND doctor_id = ?', (appointment_id, session['user_id']))
        appt = c.fetchone()
        if not appt:
            conn.close()
            return jsonify({'success': False, 'message': 'Appointment not found or unauthorized.'}), 404

        c.execute('UPDATE appointments SET status = ? WHERE id = ?', (new_status, appointment_id))
        conn.commit()
        conn.close()

        logger.info(f"Appointment {appointment_id} status updated to {new_status} by doctor {session['user_id']}")
        return jsonify({'success': True, 'message': 'Appointment status updated successfully.'}), 200
    except Exception as e:
        logger.error(f"Error updating appointment status: {e}")
        return jsonify({'success': False, 'message': 'Error updating appointment status.'}), 500

@app.route('/api/doctor-patients', methods=['GET'])
@role_required('doctor')
def doctor_patients_api():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()

        # Patients from shared reports
        c.execute('''
            SELECT u.id, u.full_name, u.email,
                   MAX(r.created_at) as last_report_date,
                   COUNT(sr.id) as total_reports_shared
            FROM shared_reports sr
            JOIN reports r ON sr.report_id = r.id
            JOIN users u ON r.user_id = u.id
            WHERE sr.doctor_id = ?
            GROUP BY u.id, u.full_name, u.email
        ''', (session['user_id'],))
        shared_rows = c.fetchall()

        # Patients from appointments
        c.execute('''
            SELECT u.id, u.full_name, u.email,
                   MAX(a.appointment_date || ' ' || a.appointment_time) as last_appointment_datetime,
                   COUNT(a.id) as total_appointments
            FROM appointments a
            JOIN users u ON a.patient_id = u.id
            WHERE a.doctor_id = ?
            GROUP BY u.id, u.full_name, u.email
        ''', (session['user_id'],))
        appt_rows = c.fetchall()

        conn.close()

        # Merge by patient id
        patients_map = {}
        for pid, name, email, last_report_date, total_shared in shared_rows:
            patients_map[pid] = {
                'id': pid,
                'name': name,
                'email': email,
                'last_report_date': last_report_date,
                'last_appointment_date': None,
                'total_reports_shared': total_shared,
                'total_appointments': 0,
            }
        for pid, name, email, last_appt_dt, total_appts in appt_rows:
            if pid not in patients_map:
                patients_map[pid] = {
                    'id': pid,
                    'name': name,
                    'email': email,
                    'last_report_date': None,
                    'last_appointment_date': last_appt_dt,
                    'total_reports_shared': 0,
                    'total_appointments': total_appts,
                }
            else:
                patients_map[pid]['last_appointment_date'] = last_appt_dt
                patients_map[pid]['total_appointments'] = total_appts

        return jsonify(list(patients_map.values())), 200
    except Exception as e:
        logger.error(f"Error fetching doctor patients: {e}")
        return jsonify({'success': False, 'message': 'Error loading patients.'}), 500

@app.route('/api/appointments', methods=['GET', 'POST'])
@role_required('patient')
def appointments_api():
    if request.method == 'GET':
        try:
            conn = sqlite3.connect('diabetic_retinopathy.db')
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            c = conn.cursor()
            c.execute('''SELECT a.id, a.appointment_date, a.appointment_time, a.reason, a.status, u.full_name as doctor_name
                        FROM appointments a
                        JOIN users u ON a.doctor_id = u.id
                        WHERE a.patient_id = ?
                        ORDER BY a.appointment_date DESC, a.appointment_time DESC''', (session['user_id'],))
            appointments_db = c.fetchall()
            conn.close()

            appointments = []
            for appt in appointments_db:
                appointments.append({
                    'id': appt[0],
                    'date': appt[1],
                    'time': appt[2],
                    'reason': appt[3],
                    'status': appt[4],
                    'doctor_name': appt[5]
                })
            return jsonify(appointments), 200
        except Exception as e:
            logger.error(f"Error fetching patient appointments: {e}")
            return jsonify({'success': False, 'message': 'Error loading appointments.'}), 500

    elif request.method == 'POST':
        data = request.get_json()
        doctor_id = data.get('doctor')
        appointment_date = data.get('date')
        appointment_time = data.get('time')
        reason = data.get('reason', '')

        if not all([doctor_id, appointment_date, appointment_time]):
            return jsonify({'success': False, 'message': 'Doctor, date, and time are required.'}), 400

        try:
            conn = sqlite3.connect('diabetic_retinopathy.db')
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            c = conn.cursor()
            c.execute('''INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, reason)
                        VALUES (?, ?, ?, ?, ?)''',
                     (session['user_id'], doctor_id, appointment_date, appointment_time, reason))
            conn.commit()
            conn.close()
            logger.info(f"Appointment scheduled for patient {session['user_id']} with doctor {doctor_id}")
            return jsonify({'success': True, 'message': 'Appointment scheduled successfully!'}), 201
        except Exception as e:
            logger.error(f"Appointment scheduling error: {e}")
            return jsonify({'success': False, 'message': 'Error scheduling appointment. Please try again.'}), 500

@app.route('/api/shared-reports', methods=['GET'])
@role_required('doctor')
def get_shared_reports_api():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        # Fetch reports shared with the logged-in doctor
        c.execute('''SELECT sr.id, r.id AS report_id, r.filename, r.predictions, r.final_prediction, r.confidence, r.created_at,
                           u_patient.full_name AS patient_name, u_patient.email AS patient_email, sr.status
                    FROM shared_reports sr
                    JOIN reports r ON sr.report_id = r.id
                    JOIN users u_patient ON r.user_id = u_patient.id
                    WHERE sr.doctor_id = ?
                    ORDER BY sr.shared_at DESC''', (session['user_id'],))
        shared_reports_db = c.fetchall()
        conn.close()

        reports = []
        for sr in shared_reports_db:
            predictions_data = json.loads(sr[3])
            # We need to find the final prediction text to display
            final_pred_idx = sr[4]
            final_pred_text = DR_STAGES.get(final_pred_idx, 'Unknown')

            reports.append({
                'id': sr[0], # shared_report_id
                'report_id': sr[1], # original report ID
                'patient_name': sr[7],
                'patient_email': sr[8],
                'date': datetime.strptime(sr[6], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),
                'result': final_pred_text,
                'confidence': f"{sr[5]:.2%}",
                'status': sr[9], # status of shared report (new, reviewed, urgent)
                'filename': sr[2]
            })
        return jsonify(reports), 200
    except Exception as e:
        logger.error(f"Error fetching shared reports: {e}")
        return jsonify({'success': False, 'message': 'Error fetching shared reports.'}), 500


@app.route('/api/shared-reports/<int:shared_report_id>/status', methods=['POST'])
@role_required('doctor')
def update_shared_report_status_api(shared_report_id):
    try:
        data = request.get_json()
        new_status = data.get('status')

        if new_status not in ['new', 'reviewed', 'urgent']:
            return jsonify({'success': False, 'message': 'Invalid status provided.'}), 400

        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()
        c.execute('UPDATE shared_reports SET status = ? WHERE id = ? AND doctor_id = ?',
                 (new_status, shared_report_id, session['user_id']))
        conn.commit()
        conn.close()

        if c.rowcount == 0:
            return jsonify({'success': False, 'message': 'Shared report not found or unauthorized.'}), 404

        logger.info(f"Shared report {shared_report_id} status updated to {new_status} by doctor {session['user_id']}")
        return jsonify({'success': True, 'message': 'Report status updated successfully.'}), 200
    except Exception as e:
        logger.error(f"Error updating shared report status: {e}")
        return jsonify({'success': False, 'message': 'Error updating report status.'}), 500


@app.route('/api/share-report', methods=['POST'])
@role_required('patient')
def share_report_api():
    data = request.get_json()
    report_id = data.get('report_id')
    doctor_id = data.get('doctor_id')

    if not all([report_id, doctor_id]):
        return jsonify({'success': False, 'message': 'Report ID and Doctor ID are required.'}), 400

    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()

        # Verify report belongs to the patient
        c.execute('SELECT user_id FROM reports WHERE id = ?', (report_id,))
        report_owner = c.fetchone()
        if not report_owner or report_owner[0] != session['user_id']:
            conn.close()
            return jsonify({'success': False, 'message': 'Report not found or not owned by patient.'}), 403

        # Verify doctor_id exists and is a doctor
        c.execute('SELECT id FROM users WHERE id = ? AND role = "doctor"', (doctor_id,))
        doctor_exists = c.fetchone()
        if not doctor_exists:
            conn.close()
            return jsonify({'success': False, 'message': 'Doctor not found.'}), 404

        # Check if already shared
        c.execute('SELECT id FROM shared_reports WHERE report_id = ? AND doctor_id = ?', (report_id, doctor_id))
        already_shared = c.fetchone()
        if already_shared:
            conn.close()
            return jsonify({'success': False, 'message': 'Report already shared with this doctor.'}), 409

        c.execute('''INSERT INTO shared_reports (report_id, doctor_id, status)
                    VALUES (?, ?, ?)''', (report_id, doctor_id, 'new'))
        conn.commit()
        conn.close()
        logger.info(f"Report {report_id} shared with doctor {doctor_id} by patient {session['user_id']}")
        return jsonify({'success': True, 'message': 'Report shared successfully.'}), 201
    except Exception as e:
        logger.error(f"Error sharing report: {e}")
        return jsonify({'success': False, 'message': 'Error sharing report.'}), 500


@app.route('/api/doctor-share-report', methods=['POST'])
@role_required('doctor')
def doctor_share_report_api():
    data = request.get_json()
    report_id = data.get('report_id')
    doctor_id = data.get('doctor_id')

    if not all([report_id, doctor_id]):
        return jsonify({'success': False, 'message': 'Report ID and Doctor ID are required.'}), 400

    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        c = conn.cursor()

        # Verify that the doctor has access to this report (it's been shared with them)
        c.execute('SELECT id FROM shared_reports WHERE report_id = ? AND doctor_id = ?', (report_id, session['user_id']))
        has_access = c.fetchone()
        if not has_access:
            conn.close()
            return jsonify({'success': False, 'message': 'Report not found or not shared with you.'}), 403

        # Verify doctor_id exists and is a doctor (but not the current doctor)
        c.execute('SELECT id FROM users WHERE id = ? AND role = "doctor"', (doctor_id,))
        doctor_exists = c.fetchone()
        if not doctor_exists:
            conn.close()
            return jsonify({'success': False, 'message': 'Doctor not found.'}), 404
        
        # Prevent doctors from sharing with themselves
        if int(doctor_id) == session['user_id']:
            conn.close()
            return jsonify({'success': False, 'message': 'You cannot share a report with yourself.'}), 400

        # Check if already shared with this doctor
        c.execute('SELECT id FROM shared_reports WHERE report_id = ? AND doctor_id = ?', (report_id, doctor_id))
        already_shared = c.fetchone()
        if already_shared:
            conn.close()
            return jsonify({'success': False, 'message': 'Report already shared with this doctor.'}), 409

        c.execute('''INSERT INTO shared_reports (report_id, doctor_id, status)
                    VALUES (?, ?, ?)''', (report_id, doctor_id, 'new'))
        conn.commit()
        conn.close()
        logger.info(f"Report {report_id} shared with doctor {doctor_id} by doctor {session['user_id']}")
        return jsonify({'success': True, 'message': 'Report shared successfully.'}), 201
    except Exception as e:
        logger.error(f"Error sharing report: {e}")
        return jsonify({'success': False, 'message': 'Error sharing report.'}), 500

# Important: Make sure to serve static files (like uploaded images)
# The '/uploads/<filename>' route already handles serving uploaded images
# If you have other static files (e.g., in a 'static' folder for HTML templates), Flask automatically serves them.

if __name__ == '__main__':
    try:
        init_db()
        models_dict = load_all_models()  # Update global models_dict
        if not models_dict:
            logger.critical("No models loaded successfully. Application may not function as expected.")
        logger.info("Application started successfully")
        app.run(debug=True, host='0.0.0.0', port=5000) # Set debug=False for production
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        raise