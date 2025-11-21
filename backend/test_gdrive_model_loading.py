#!/usr/bin/env python3
"""
Test script to verify that models can be loaded directly from Google Drive
"""

import os
import sys
import logging

# Add the parent directory to sys.path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gdrive_model_loading():
    """Test that models can be loaded directly from Google Drive"""
    try:
        # Import the app module
        import app
        
        # Test loading stage 1 model
        logger.info("Testing stage 1 model loading from Google Drive...")
        stage1_model = app.load_stage1_model()
        if stage1_model is not None:
            logger.info("✓ Stage 1 model loaded successfully from Google Drive")
        else:
            logger.error("✗ Failed to load stage 1 model from Google Drive")
            return False
        
        # Test loading clinical model
        logger.info("Testing clinical model loading from Google Drive...")
        clinical_model = app.load_clinical_model()
        if clinical_model is not None:
            logger.info("✓ Clinical model loaded successfully from Google Drive")
        else:
            logger.error("✗ Failed to load clinical model from Google Drive")
            return False
            
        # Test loading all models
        logger.info("Testing all models loading from Google Drive...")
        models_dict = app.load_all_models()
        
        # Check that we have models loaded
        loaded_count = sum(1 for model_data in models_dict.values() if model_data[0] is not None)
        total_count = len(models_dict)
        
        logger.info(f"Models loaded: {loaded_count}/{total_count}")
        
        if loaded_count > 0:
            logger.info("✓ Google Drive model loading test PASSED")
            return True
        else:
            logger.error("✗ Google Drive model loading test FAILED - no models loaded")
            return False
            
    except Exception as e:
        logger.error(f"Google Drive model loading test FAILED with exception: {e}")
        return False

def main():
    """Main function"""
    print("Google Drive Model Loading Test")
    print("=" * 35)
    
    success = test_gdrive_model_loading()
    
    if success:
        print("\n✓ Google Drive model loading test passed!")
        print("The application can now load models directly from Google Drive.")
        return 0
    else:
        print("\n✗ Google Drive model loading test failed!")
        print("Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())