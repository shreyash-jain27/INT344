# ============================================
# src/multilingual_processor.py
# Process text in multiple Indian languages
# ============================================

import re
from typing import List, Dict, Optional
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize import indic_normalize
from indicnlp import common

# Set IndicNLP data path
try:
    common.set_resources_path("indicnlp_resources")
except:
    pass


class MultilingualProcessor:
    """
    Process text in multiple Indian languages
    Handles tokenization, normalization, and text cleaning
    """
    
    # Language-specific settings
    LANGUAGE_CONFIG = {
        'hi': {'name': 'Hindi', 'code': 'hin'},
        'ta': {'name': 'Tamil', 'code': 'tam'},
        'te': {'name': 'Telugu', 'code': 'tel'},
        'bn': {'name': 'Bengali', 'code': 'ben'},
        'mr': {'name': 'Marathi', 'code': 'mar'},
        'gu': {'name': 'Gujarati', 'code': 'guj'},
        'kn': {'name': 'Kannada', 'code': 'kan'},
        'ml': {'name': 'Malayalam', 'code': 'mal'},
        'pa': {'name': 'Punjabi', 'code': 'pan'},
        'or': {'name': 'Odia', 'code': 'ori'},
        'as': {'name': 'Assamese', 'code': 'asm'}
    }
    
    def __init__(self):
        self.normalizer_factory = indic_normalize.IndicNormalizerFactory()
    
    def normalize_text(self, text: str, lang: str) -> str:
        """
        Normalize Indian language text
        Handles various scripts and special characters
        """
        if lang not in self.LANGUAGE_CONFIG or lang == 'en':
            return text
        
        try:
            # Get normalizer for language
            normalizer = self.normalizer_factory.get_normalizer(lang)
            
            # Normalize
            normalized = normalizer.normalize(text)
            
            return normalized
        except Exception as e:
            print(f"⚠️ Normalization failed for {lang}: {e}")
            return text
    
    def tokenize_sentences(self, text: str, lang: str) -> List[str]:
        """
        Tokenize text into sentences for Indian languages
        """
        if not text:
            return []
        
        # For English, use simple splitting
        if lang == 'en':
            import nltk
            try:
                return nltk.sent_tokenize(text)
            except:
                # Fallback to regex
                return re.split(r'[।.!?]+\s+', text)
        
        # For Indian languages
        try:
            lang_code = self.LANGUAGE_CONFIG.get(lang, {}).get('code', lang)
            sentences = sentence_tokenize.sentence_split(text, lang=lang_code)
            return sentences
        except Exception as e:
            print(f"⚠️ Sentence tokenization failed for {lang}: {e}")
            # Fallback: split by Indian sentence terminator (।) and period
            return re.split(r'[।.!?]+\s+', text)
    
    def tokenize_words(self, text: str, lang: str) -> List[str]:
        """
        Tokenize text into words for Indian languages
        """
        if not text:
            return []
        
        # For English
        if lang == 'en':
            return text.split()
        
        # For Indian languages
        try:
            lang_code = self.LANGUAGE_CONFIG.get(lang, {}).get('code', lang)
            words = indic_tokenize.trivial_tokenize(text, lang=lang_code)
            return words
        except Exception as e:
            print(f"⚠️ Word tokenization failed for {lang}: {e}")
            # Fallback to whitespace splitting
            return text.split()
    
    def remove_stopwords(self, words: List[str], lang: str) -> List[str]:
        """
        Remove stopwords for Indian languages
        """
        # Basic stopwords for major Indian languages
        stopwords = {
            'hi': {'है', 'हैं', 'था', 'थे', 'की', 'का', 'के', 'में', 'से', 'को', 'ने', 'और', 'या', 'एक', 'यह', 'वह'},
            'ta': {'உள்ள', 'என்று', 'ஆகும்', 'மற்றும்', 'அல்லது', 'இது', 'அது', 'ஒரு'},
            'te': {'ఉంది', 'ఉన్న', 'అని', 'మరియు', 'లేదా', 'ఇది', 'అది', 'ఒక'},
            'bn': {'আছে', 'ছিল', 'এবং', 'বা', 'এক', 'এই', 'সেই', 'যে'},
            'pa': {'ਹੈ', 'ਹਨ', 'ਸੀ', 'ਸਨ', 'ਦੀ', 'ਦਾ', 'ਦੇ', 'ਵਿੱਚ', 'ਤੋਂ', 'ਨੂੰ', 'ਨੇ', 'ਅਤੇ', 'ਜਾਂ', 'ਇੱਕ', 'ਇਹ', 'ਉਹ'},
            'en': {'the', 'is', 'are', 'was', 'were', 'in', 'of', 'to', 'and', 'or', 'a', 'an'}
        }
        
        lang_stopwords = stopwords.get(lang, set())
        
        if not lang_stopwords:
            return words
        
        return [w for w in words if w.lower() not in lang_stopwords]
    
    def preprocess_for_ml(self, text: str, lang: str) -> str:
        """
        Preprocess text for machine learning models
        """
        # Normalize
        text = self.normalize_text(text, lang)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep language scripts)
        # Remove only ASCII special chars
        text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]', ' ', text)
        
        # Clean up
        text = ' '.join(text.split())
        
        return text
    
    def split_code_mixed_text(self, text: str) -> Dict[str, List[str]]:
        """
        Split code-mixed text into language segments
        E.g., Hinglish -> Hindi segments + English segments
        """
        # Simple approach: detect script changes
        segments = {'script_changes': []}
        
        current_script = None
        current_segment = []
        
        for char in text:
            code_point = ord(char)
            
            # Detect script
            if 0x0900 <= code_point <= 0x097F:  # Devanagari
                script = 'hi'
            elif 0x0B80 <= code_point <= 0x0BFF:  # Tamil
                script = 'ta'
            elif 0x0C00 <= code_point <= 0x0C7F:  # Telugu
                script = 'te'
            elif 0x0041 <= code_point <= 0x007A or 0x0061 <= code_point <= 0x007A:  # Latin
                script = 'en'
            elif char.isspace():
                current_segment.append(char)
                continue
            else:
                script = 'other'
            
            if current_script is None:
                current_script = script
            
            if script != current_script:
                if current_segment:
                    segments['script_changes'].append({
                        'script': current_script,
                        'text': ''.join(current_segment).strip()
                    })
                current_script = script
                current_segment = [char]
            else:
                current_segment.append(char)
        
        # Add last segment
        if current_segment:
            segments['script_changes'].append({
                'script': current_script,
                'text': ''.join(current_segment).strip()
            })
        
        return segments
    
    def get_text_statistics(self, text: str, lang: str) -> Dict:
        """
        Get statistics about text in given language
        """
        sentences = self.tokenize_sentences(text, lang)
        words = self.tokenize_words(text, lang)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'language': lang
        }
    
    def prepare_for_translation(self, text: str, lang: str) -> str:
        """
        Prepare text for translation
        Clean and normalize while preserving meaning
        """
        # Normalize
        text = self.normalize_text(text, lang)
        
        # Tokenize and rejoin to handle spacing issues
        sentences = self.tokenize_sentences(text, lang)
        
        # Clean each sentence
        cleaned_sentences = []
        for sent in sentences:
            # Remove excessive punctuation
            sent = re.sub(r'([।.!?]){2,}', r'\1', sent)
            # Normalize whitespace
            sent = ' '.join(sent.split())
            if sent:
                cleaned_sentences.append(sent)
        
        return ' '.join(cleaned_sentences)


# ============================================
# Utility functions
# ============================================

def quick_tokenize(text: str, lang: str = 'en') -> List[str]:
    """Quick tokenization utility"""
    processor = MultilingualProcessor()
    return processor.tokenize_sentences(text, lang)


def quick_normalize(text: str, lang: str) -> str:
    """Quick normalization utility"""
    processor = MultilingualProcessor()
    return processor.normalize_text(text, lang)


# ============================================
# Test function
# ============================================

def test_multilingual_processing():
    """Test multilingual processing"""
    processor = MultilingualProcessor()
    
    samples = {
        'hi': 'कंपनी ने इस वर्ष रिकॉर्ड राजस्व हासिल किया। यह पिछले साल से 25% अधिक है।',
        'ta': 'நிறுவனம் இந்த ஆண்டு சாதனை வருவாயை அடைந்தது। இது கடந்த ஆண்டை விட 25% அதிகம்.',
        'en': 'The company achieved record revenue this year. This is 25% more than last year.'
    }
    
    print("Testing multilingual processing:\n")
    
    for lang, text in samples.items():
        print(f"{lang.upper()}:")
        print(f"Original: {text}")
        
        # Normalize
        normalized = processor.normalize_text(text, lang)
        print(f"Normalized: {normalized}")
        
        # Tokenize sentences
        sentences = processor.tokenize_sentences(text, lang)
        print(f"Sentences: {len(sentences)}")
        
        # Tokenize words
        words = processor.tokenize_words(text, lang)
        print(f"Words: {len(words)}")
        
        # Statistics
        stats = processor.get_text_statistics(text, lang)
        print(f"Stats: {stats}")
        print()


if __name__ == "__main__":
    test_multilingual_processing()