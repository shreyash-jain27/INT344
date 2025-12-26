# ============================================
# src/language_detector.py
# Advanced language detection for Indian languages
# ============================================

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import langdetect
from langdetect import detect, detect_langs, LangDetectException

# Try to import pycld2 for better detection
try:
    import pycld2 as cld2
    CLD2_AVAILABLE = True
except:
    CLD2_AVAILABLE = False
    print("⚠️ pycld2 not available, using langdetect only")


class LanguageDetector:
    """
    Advanced language detection system for Indian languages
    Handles code-mixed text and provides confidence scores
    """
    
    # Indian language information
    INDIAN_LANGUAGES = {
        'hi': {
            'name': 'Hindi',
            'native': 'हिन्दी',
            'script': 'Devanagari',
            'unicode_range': (0x0900, 0x097F)
        },
        'ta': {
            'name': 'Tamil',
            'native': 'தமிழ்',
            'script': 'Tamil',
            'unicode_range': (0x0B80, 0x0BFF)
        },
        'te': {
            'name': 'Telugu',
            'native': 'తెలుగు',
            'script': 'Telugu',
            'unicode_range': (0x0C00, 0x0C7F)
        },
        'bn': {
            'name': 'Bengali',
            'native': 'বাংলা',
            'script': 'Bengali',
            'unicode_range': (0x0980, 0x09FF)
        },
        'mr': {
            'name': 'Marathi',
            'native': 'मराठी',
            'script': 'Devanagari',
            'unicode_range': (0x0900, 0x097F)
        },
        'gu': {
            'name': 'Gujarati',
            'native': 'ગુજરાતી',
            'script': 'Gujarati',
            'unicode_range': (0x0A80, 0x0AFF)
        },
        'kn': {
            'name': 'Kannada',
            'native': 'ಕನ್ನಡ',
            'script': 'Kannada',
            'unicode_range': (0x0C80, 0x0CFF)
        },
        'ml': {
            'name': 'Malayalam',
            'native': 'മലയാളം',
            'script': 'Malayalam',
            'unicode_range': (0x0D00, 0x0D7F)
        },
        'pa': {
            'name': 'Punjabi',
            'native': 'ਪੰਜਾਬੀ',
            'script': 'Gurmukhi',
            'unicode_range': (0x0A00, 0x0A7F)
        },
        'or': {
            'name': 'Odia',
            'native': 'ଓଡ଼ିଆ',
            'script': 'Odia',
            'unicode_range': (0x0B00, 0x0B7F)
        },
        'as': {
            'name': 'Assamese',
            'native': 'অসমীয়া',
            'script': 'Bengali',
            'unicode_range': (0x0980, 0x09FF)
        },
        'en': {
            'name': 'English',
            'native': 'English',
            'script': 'Latin',
            'unicode_range': (0x0000, 0x007F)
        }
    }
    
    def __init__(self):
        # Set seed for langdetect (for reproducibility)
        langdetect.DetectorFactory.seed = 0
    
    def detect_language(
        self,
        text: str,
        return_all: bool = False
    ) -> Dict[str, any]:
        """
        Detect language(s) in text
        
        Args:
            text: Input text
            return_all: Return all detected languages with probabilities
            
        Returns:
            Dictionary with language info
        """
        if not text or len(text.strip()) < 10:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'is_code_mixed': False
            }
        
        # Clean text
        clean_text = self._clean_text(text)
        
        # Detect script first (more reliable for Indian languages)
        script_detection = self._detect_by_script(clean_text)
        
        # Use langdetect
        langdetect_result = self._detect_with_langdetect(clean_text)
        
        # Use CLD2 if available
        if CLD2_AVAILABLE:
            cld2_result = self._detect_with_cld2(clean_text)
        else:
            cld2_result = None
        
        # Combine results
        final_result = self._combine_detections(
            script_detection,
            langdetect_result,
            cld2_result
        )
        
        # Check for code-mixing
        final_result['is_code_mixed'] = self._is_code_mixed(clean_text)
        
        # Get all languages if requested
        if return_all:
            final_result['all_languages'] = self._get_all_languages(clean_text)
        
        return final_result
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better detection"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _detect_by_script(self, text: str) -> Dict[str, float]:
        """
        Detect language by Unicode script ranges
        Most reliable for Indian languages
        """
        script_counts = {}
        
        for char in text:
            code_point = ord(char)
            
            for lang_code, info in self.INDIAN_LANGUAGES.items():
                start, end = info['unicode_range']
                if start <= code_point <= end:
                    script_counts[lang_code] = script_counts.get(lang_code, 0) + 1
        
        if not script_counts:
            return {}
        
        total = sum(script_counts.values())
        script_probs = {
            lang: count / total
            for lang, count in script_counts.items()
        }
        
        return script_probs
    
    def _detect_with_langdetect(self, text: str) -> Optional[Dict]:
        """Use langdetect library"""
        try:
            # Get primary language
            primary = detect(text)
            
            # Get all with probabilities
            langs = detect_langs(text)
            
            return {
                'primary': primary,
                'all': {str(lang).split(':')[0]: float(str(lang).split(':')[1]) 
                       for lang in langs}
            }
        except LangDetectException:
            return None
    
    def _detect_with_cld2(self, text: str) -> Optional[Dict]:
        """Use CLD2 for detection (more accurate but requires compilation)"""
        if not CLD2_AVAILABLE:
            return None
        
        try:
            is_reliable, text_bytes_found, details = cld2.detect(text)
            
            if details:
                return {
                    'primary': details[0][1],
                    'confidence': details[0][2] / 100.0,
                    'reliable': is_reliable
                }
        except:
            return None
    
    def _combine_detections(
        self,
        script_result: Dict,
        langdetect_result: Optional[Dict],
        cld2_result: Optional[Dict]
    ) -> Dict:
        """Combine results from multiple detectors"""
        
        # Priority: Script detection (most reliable for Indian languages)
        if script_result:
            primary_lang = max(script_result.items(), key=lambda x: x[1])
            return {
                'language': primary_lang[0],
                'confidence': primary_lang[1],
                'method': 'script',
                'script_probs': script_result
            }
        
        # Fallback to langdetect
        if langdetect_result:
            return {
                'language': langdetect_result['primary'],
                'confidence': langdetect_result['all'].get(
                    langdetect_result['primary'], 0.0
                ),
                'method': 'langdetect',
                'all_langs': langdetect_result['all']
            }
        
        # Fallback to CLD2
        if cld2_result:
            return {
                'language': cld2_result['primary'],
                'confidence': cld2_result['confidence'],
                'method': 'cld2'
            }
        
        # Unknown
        return {
            'language': 'en',  # Default to English
            'confidence': 0.5,
            'method': 'default'
        }
    
    def _is_code_mixed(self, text: str) -> bool:
        """
        Detect if text is code-mixed (e.g., Hinglish)
        """
        script_probs = self._detect_by_script(text)
        
        if not script_probs:
            return False
        
        # If multiple scripts with significant presence
        significant_scripts = [
            lang for lang, prob in script_probs.items()
            if prob > 0.15  # At least 15% of characters
        ]
        
        return len(significant_scripts) > 1
    
    def _get_all_languages(self, text: str) -> List[Tuple[str, float]]:
        """Get all detected languages with probabilities"""
        script_probs = self._detect_by_script(text)
        
        if not script_probs:
            return [('en', 1.0)]
        
        # Sort by probability
        return sorted(
            script_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def get_language_info(self, lang_code: str) -> Dict:
        """Get detailed information about a language"""
        return self.INDIAN_LANGUAGES.get(lang_code, {
            'name': 'Unknown',
            'native': 'Unknown',
            'script': 'Unknown'
        })
    
    def detect_document_language(
        self,
        sections: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Detect language for each section of a document
        
        Args:
            sections: Dict of section_name -> text
            
        Returns:
            Dict of section_name -> language_info
        """
        results = {}
        
        print("\n🌍 Detecting languages in document sections...")
        
        for section_name, text in sections.items():
            if not text or len(text.strip()) < 20:
                continue
            
            detection = self.detect_language(text)
            lang_info = self.get_language_info(detection['language'])
            
            results[section_name] = {
                'language': detection['language'],
                'language_name': lang_info['name'],
                'script': lang_info['script'],
                'confidence': detection['confidence'],
                'is_code_mixed': detection.get('is_code_mixed', False)
            }
            
            print(f"  {section_name}: {lang_info['name']} "
                  f"({detection['confidence']*100:.1f}% confidence)")
            
            if detection.get('is_code_mixed'):
                print(f"    ⚠️ Code-mixed text detected")
        
        return results
    
    def get_document_primary_language(
        self,
        sections: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        Get the primary language of entire document
        
        Returns:
            (language_code, confidence)
        """
        # Combine all text
        full_text = ' '.join(sections.values())
        
        detection = self.detect_language(full_text)
        
        return detection['language'], detection['confidence']


# ============================================
# Utility functions
# ============================================

def detect_text_language(text: str) -> str:
    """Quick utility to detect language"""
    detector = LanguageDetector()
    result = detector.detect_language(text)
    return result['language']


def is_indian_language(lang_code: str) -> bool:
    """Check if language code is an Indian language"""
    indian_langs = ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as']
    return lang_code in indian_langs


# ============================================
# Test function
# ============================================

def test_language_detection():
    """Test language detection with samples"""
    detector = LanguageDetector()
    
    samples = {
        'English': 'The company achieved record revenue this year.',
        'Hindi': 'कंपनी ने इस वर्ष रिकॉर्ड राजस्व हासिल किया।',
        'Tamil': 'நிறுவனம் இந்த ஆண்டு சாதனை வருவாயை அடைந்தது.',
        'Telugu': 'కంపెనీ ఈ సంవత్సరం రికార్డు రాబడిని సాధించింది.',
        'Hinglish': 'Company ne is year record revenue achieve kiya hai.'
    }
    
    print("Testing language detection:\n")
    
    for name, text in samples.items():
        result = detector.detect_language(text, return_all=True)
        print(f"{name}:")
        print(f"  Detected: {result['language']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Code-mixed: {result['is_code_mixed']}")
        if 'all_languages' in result:
            print(f"  All languages: {result['all_languages']}")
        print()


if __name__ == "__main__":
    test_language_detection()