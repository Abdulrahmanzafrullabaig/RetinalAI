import os
import gdown
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Google Drive file IDs and target paths
MODEL_DOWNLOADS = [
    {
        'name': 'Clinical Prediction Model',
        'url': 'https://drive.google.com/uc?id=18tJ7d4BbdWReVUCWSvfF_cY3yMh46PZI',
        'output': 'models/dr_model.pth'
    },
    {
        'name': 'Stage 1 Fundus Classifier',
        'url': 'https://drive.google.com/uc?id=1sWu9cAiz7Z2DqnEsXllpqOcMToBDF2Z7',
        'output': 'models/fundus_classifier.pth'
    },
    {
        'name': 'ResNet50 Model',
        'url': 'https://drive.google.com/uc?id=1gFACSGC_7PON0xZoII6PJ5I9duk9ePGs',
        'output': 'models/model1.pth'
    },
    {
        'name': 'EfficientNet Model',
        'url': 'https://drive.google.com/uc?id=12codw5Zjq-ShLuhd_CUE_ZWf8DKUsRd0',
        'output': 'models/model2.pth'
    },
    {
        'name': 'VGG16 Model',
        'url': 'https://drive.google.com/uc?id=1MhQVHXSKMOTWItuau-Idjn4aSNQhACQJ',
        'output': 'models/model3.pth'
    },
    {
        'name': 'DenseNet121 Model',
        'url': 'https://drive.google.com/uc?id=1TPTcVIZx0NPmHypNuoIMCop8kkgxQgZZ',
        'output': 'models/model4.pth'
    }
]

def download_models():
    """Download all models from Google Drive"""
    # Create models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created directory: {models_dir}")
    
    # Download each model
    for model_info in MODEL_DOWNLOADS:
        try:
            logger.info(f"Downloading {model_info['name']}...")
            gdown.download(model_info['url'], model_info['output'], quiet=False)
            logger.info(f"Successfully downloaded {model_info['name']} to {model_info['output']}")
        except Exception as e:
            logger.error(f"Failed to download {model_info['name']}: {e}")

if __name__ == "__main__":
    download_models()