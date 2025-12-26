# ============================================
# src/ocr_processor.py
# OCR support for scanned PDFs
# ============================================

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

try:
    from .ocr_config import OCRConfig
    OCRConfig.setup_tesseract()
except:
    pass


class OCRProcessor:
    """
    Handle OCR for scanned PDFs
    Supports both Tesseract and EasyOCR
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.easyocr_reader = None
        
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based)
        Returns True if scanned, False if native text
        """
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                # Check first 3 pages
                for page in pdf.pages[:3]:
                    text = page.extract_text()
                    
                    # If we get substantial text, it's not scanned
                    if text and len(text.strip()) > 100:
                        return False
                
                # If no text found, likely scanned
                return True
                
        except Exception as e:
            print(f"Error checking PDF type: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results
        """
        # Convert PIL to OpenCV format
        img = np.array(image)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Threshold
        _, thresh = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return thresh
    
    def ocr_with_tesseract(
        self,
        image: Image.Image,
        lang: str = 'eng',
        preprocess: bool = True
    ) -> Tuple[str, float]:
        """
        Perform OCR using Tesseract
        Returns (text, confidence)
        """
        try:
            if preprocess:
                processed = self.preprocess_image(image)
                image = Image.fromarray(processed)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if conf > 0:  # Valid detection
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return full_text, avg_confidence
            
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return "", 0.0
    
    def ocr_with_easyocr(
        self,
        image: Image.Image,
        languages: List[str] = ['en', 'hi']
    ) -> Tuple[str, float]:
        """
        Perform OCR using EasyOCR
        Returns (text, confidence)
        """
        try:
            import easyocr
            # Initialize reader if not already done
            if self.easyocr_reader is None:
                print("Loading EasyOCR models (first time only)...")
                self.easyocr_reader = easyocr.Reader(
                    languages,
                    gpu=self.use_gpu
                )
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            results = self.easyocr_reader.readtext(img_array)
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for detection in results:
                bbox, text, conf = detection
                texts.append(text)
                confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return full_text, avg_confidence
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0.0
    
    def ocr_pdf(
        self,
        pdf_path: str,
        method: str = 'auto',
        languages: List[str] = ['eng']
    ) -> Dict[int, Dict]:
        """
        Perform OCR on entire PDF
        
        Args:
            pdf_path: Path to PDF file
            method: 'tesseract', 'easyocr', or 'auto'
            languages: List of language codes
            
        Returns:
            Dictionary mapping page numbers to OCR results
        """
        print(f"\n📄 Starting OCR on PDF: {Path(pdf_path).name}")
        print(f"Method: {method}, Languages: {languages}")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        try:
            images = convert_from_path(pdf_path, dpi=300)
            print(f"✓ Converted {len(images)} pages")
        except Exception as e:
            print(f"❌ Error converting PDF: {e}")
            return {}
        
        # Perform OCR on each page
        results = {}
        
        for page_num, image in enumerate(images, 1):
            print(f"\n  Processing page {page_num}/{len(images)}...")
            
            # Choose OCR method
            if method == 'auto':
                # Try Tesseract first (faster)
                text, conf = self.ocr_with_tesseract(image, lang=languages[0])
                
                # If confidence is low, try EasyOCR
                if conf < 70 and len(languages) > 1:
                    print(f"    Low confidence ({conf:.1f}%), trying EasyOCR...")
                    text2, conf2 = self.ocr_with_easyocr(image, languages)
                    
                    if conf2 > conf:
                        text, conf = text2, conf2
                        print(f"    ✓ EasyOCR better ({conf2:.1f}%)")
                    
            elif method == 'tesseract':
                text, conf = self.ocr_with_tesseract(image, lang=languages[0])
                
            elif method == 'easyocr':
                text, conf = self.ocr_with_easyocr(image, languages)
            
            results[page_num] = {
                'text': text,
                'confidence': conf,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
            print(f"    ✓ Extracted {len(text.split())} words (confidence: {conf:.1f}%)")
        
        # Summary
        total_words = sum(r['word_count'] for r in results.values())
        avg_conf = sum(r['confidence'] for r in results.values()) / len(results)
        
        print(f"\n✅ OCR Complete!")
        print(f"   Total pages: {len(results)}")
        print(f"   Total words: {total_words:,}")
        print(f"   Average confidence: {avg_conf:.1f}%")
        
        return results
    
    def combine_with_native_extraction(
        self,
        pdf_path: str,
        native_text: str
    ) -> str:
        """
        Combine native PDF extraction with OCR for hybrid PDFs
        """
        # Check if native extraction is sufficient
        if len(native_text.strip()) > 1000:
            print("✓ Using native PDF extraction (sufficient text found)")
            return native_text
        
        # Check if PDF needs OCR
        if self.is_scanned_pdf(pdf_path):
            print("⚠️ PDF appears to be scanned, applying OCR...")
            ocr_results = self.ocr_pdf(pdf_path, method='auto')
            
            # Combine OCR text
            ocr_text = '\n\n'.join(
                f"=== PAGE {page} ===\n{data['text']}"
                for page, data in ocr_results.items()
            )
            
            return ocr_text
        
        return native_text


# ============================================
# Utility functions
# ============================================

def detect_language_from_text(text: str) -> str:
    """Detect language of text"""
    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except:
        return 'en'


def extract_with_ocr_fallback(pdf_path: str) -> str:
    """
    Smart extraction: try native first, fall back to OCR
    """
    import pdfplumber
    
    print(f"\n📄 Extracting from: {Path(pdf_path).name}")
    
    # Try native extraction
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        
        # Check if extraction was successful
        if len(text.strip()) > 500:
            print("✓ Native extraction successful")
            return text
    except Exception as e:
        print(f"Native extraction failed: {e}")
    
    # Fall back to OCR
    print("Falling back to OCR...")
    ocr_processor = OCRProcessor()
    
    if ocr_processor.is_scanned_pdf(pdf_path):
        results = ocr_processor.ocr_pdf(pdf_path)
        text = '\n\n'.join(
            f"=== PAGE {page} ===\n{data['text']}"
            for page, data in results.items()
        )
        return text
    
    return ""