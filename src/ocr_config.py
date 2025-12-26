# ============================================
# src/ocr_config.py
# Configuration for OCR engines
# ============================================

import os
from pathlib import Path

class OCRConfig:
    """Configuration for OCR tools"""
    
    # Tesseract path (Windows)
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Check if Tesseract exists
    @staticmethod
    def setup_tesseract():
        """Setup Tesseract path"""
        try:
            import pytesseract
            
            # Try default path first
            if os.path.exists(OCRConfig.TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = OCRConfig.TESSERACT_PATH
                return True
            
            # Try common alternative paths
            alternative_paths = [
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
                "/usr/bin/tesseract",  # Linux
                "/usr/local/bin/tesseract"  # Mac
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✓ Tesseract found at: {path}")
                    return True
            
            print("⚠️ Tesseract not found. OCR features will be limited.")
            return False
            
        except ImportError:
            print("⚠️ pytesseract not installed")
            return False
    
    # Supported languages for OCR
    SUPPORTED_LANGUAGES = {
        'eng': 'English',
        'hin': 'Hindi',
        'tam': 'Tamil',
        'tel': 'Telugu',
        'ben': 'Bengali',
        'guj': 'Gujarati',
        'kan': 'Kannada',
        'mal': 'Malayalam',
        'mar': 'Marathi',
        'pan': 'Punjabi'
    }
    
    # OCR confidence threshold
    MIN_CONFIDENCE = 60.0  # Minimum confidence to accept OCR result
    
    # EasyOCR configuration
    EASYOCR_LANGUAGES = ['en', 'hi']  # English and Hindi
    EASYOCR_GPU = True  # Use GPU if available


# Test function
if __name__ == "__main__":
    config = OCRConfig()
    if config.setup_tesseract():
        print("✅ OCR configuration successful")
    else:
        print("❌ OCR configuration failed")