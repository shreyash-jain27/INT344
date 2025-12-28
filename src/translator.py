# ============================================
# src/translator.py
# Translation system for Indian languages
# ============================================

import time
from typing import Dict, List, Optional, Tuple
from deep_translator import GoogleTranslator

try:
    from .language_detector import LanguageDetector
except:
    from language_detector import LanguageDetector


class MultilingualTranslator:
    """
    Translation system supporting Indian languages
    Uses Google Translate API and transformer models
    """
    
    # Language mappings
    LANGUAGE_NAMES = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'or': 'Odia',
        'as': 'Assamese'
    }
    
    # Google Translate language codes
    GOOGLE_CODES = {
        'hi': 'hi', 'ta': 'ta', 'te': 'te', 'bn': 'bn',
        'mr': 'mr', 'gu': 'gu', 'kn': 'kn', 'ml': 'ml',
        'pa': 'pa', 'or': 'or', 'as': 'as', 'en': 'en'
    }
    
    def __init__(self, use_gpu: bool = True):
        import torch
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.language_detector = LanguageDetector()
        
        # Lazy load models
        self.indic_model = None
        self.indic_tokenizer = None
        
        # Local model for Punjabi
        self.punjabi_model = None
        self.punjabi_tokenizer = None
        self.local_model_path = "models/punjabi_translator"
        
        # Cache for translations
        self.translation_cache = {}
    
    def _load_indic_model(self):
        """Load IndicBART model for better Indian language translation"""
        if self.indic_model is not None:
            return
            
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        print("\n🤖 Loading IndicBART for translation...")
        print("   This may take a few minutes on first use...")
        
        try:
            model_name = "ai4bharat/IndicBARTSS"
            
            self.indic_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                use_fast=False,
                keep_accents=True
            )
            
            self.indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_safetensors=True
            )
            
            if self.use_gpu:
                self.indic_model = self.indic_model.to('cuda')
            
            print("   ✓ IndicBART loaded successfully!")
            
        except Exception as e:
            print(f"   ⚠️ Could not load IndicBART: {e}")
            if "device-side assert" in str(e).lower():
                print("   🚨 CUDA Context Corrupted. Forcing CPU usage for safety.")
                self.use_gpu = False
            print("   Will use Google Translate as fallback")
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        method: str = 'auto'
    ) -> Dict:
        """
        Translate text between languages
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            method: 'google', 'indicbart', or 'auto'
            
        Returns:
            Dict with translation and metadata
        """
        if not text or len(text.strip()) < 3:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'none',
                'success': True
            }
        
        # Same language - no translation needed
        if source_lang == target_lang:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'none',
                'success': True
            }
        
        # Check cache
        cache_key = f"{source_lang}_{target_lang}_{hash(text)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Choose method
        if method == 'auto':
            # Check if local Punjabi model exists and target is Punjabi
            import os
            has_local_punjabi = os.path.exists(os.path.join(self.local_model_path, "config.json"))
            
            if target_lang == 'pa' and has_local_punjabi:
                method = 'local_punjabi'
            # Use IndicBART for Indian language pairs
            elif source_lang in self.GOOGLE_CODES and target_lang in self.GOOGLE_CODES:
                if source_lang != 'en' or target_lang != 'en':
                    method = 'indicbart'
                else:
                    method = 'google'
            else:
                method = 'google'
        
        # Perform translation
        start_time = time.time()
        
        if method == 'indicbart':
            result = self._translate_with_indicbart(text, source_lang, target_lang)
        elif method == 'local_punjabi':
            result = self._translate_with_local_punjabi(text, source_lang, target_lang)
        else:
            result = self._translate_with_google(text, source_lang, target_lang)
        
        result['time'] = time.time() - start_time
        
        # Cache result
        self.translation_cache[cache_key] = result
        
        return result
    
    def _translate_with_google(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict:
        """Translate using Google Translate"""
        try:
            # Map to Google language codes
            source_code = self.GOOGLE_CODES.get(source_lang, source_lang)
            target_code = self.GOOGLE_CODES.get(target_lang, target_lang)
            
            # Split into chunks if too long (Google has 5000 char limit)
            chunks = self._split_into_chunks(text, max_length=4000)
            
            translated_chunks = []
            for chunk in chunks:
                translator = GoogleTranslator(
                    source=source_code,
                    target=target_code
                )
                translated = translator.translate(chunk)
                translated_chunks.append(translated)
            
            translated_text = ' '.join(translated_chunks)
            
            return {
                'translated_text': translated_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'google',
                'success': True,
                'chunks': len(chunks)
            }
            
        except Exception as e:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'google',
                'success': False,
                'error': str(e)
            }
    
    def _translate_with_indicbart(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict:
        """Translate using IndicBART model"""
        try:
            # Load model if needed
            self._load_indic_model()
            
            if self.indic_model is None:
                # Fallback to Google
                return self._translate_with_google(text, source_lang, target_lang)
            
            # Prepare input
            input_text = f"<{source_lang}> {text} </{source_lang}> <{target_lang}>"
            
            inputs = self.indic_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.use_gpu:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate translation
            import torch
            with torch.no_grad():
                generated_tokens = self.indic_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # Decode
            translated_text = self.indic_tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True
            )
            
            return {
                'translated_text': translated_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'indicbart',
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ IndicBART translation failed: {e}")
            # Fallback to Google
            return self._translate_with_google(text, source_lang, target_lang)

    def _load_local_punjabi_model(self):
        """Load locally trained Punjabi translation model"""
        if self.punjabi_model is not None:
            return
        
        print(f"\n🤖 Loading local Punjabi translation model from {self.local_model_path}...")
        
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.punjabi_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.punjabi_model = AutoModelForSeq2SeqLM.from_pretrained(self.local_model_path)
            
            if self.use_gpu:
                try:
                    self.punjabi_model = self.punjabi_model.to('cuda')
                except Exception as cuda_e:
                    print(f"   ⚠️ Failed to move to GPU: {cuda_e}")
                    self.punjabi_model = self.punjabi_model.to('cpu')
            
            print("   ✓ Local Punjabi model loaded successfully!")
            
        except Exception as e:
            print(f"   ⚠️ Could not load local Punjabi model: {e}")
            if "device-side assert" in str(e).lower():
                self.use_gpu = False
            print("   Will use IndicBART or Google Translate as fallback")

    def _translate_with_local_punjabi(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict:
        """Translate using local Punjabi model"""
        try:
            # Load model if needed
            self._load_local_punjabi_model()
            
            if self.punjabi_model is None:
                # Fallback to IndicBART
                return self._translate_with_indicbart(text, source_lang, target_lang)
            
            # Prepare input
            # Helsinki-NLP group models (en-inc) require a language tag
            tagged_text = f">>pan<< {text}"
            
            inputs = self.punjabi_tokenizer(
                tagged_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.use_gpu:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                try:
                    generated_tokens = self.punjabi_model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=5,
                        num_return_sequences=1,
                        early_stopping=True
                    )
                except Exception as gen_e:
                    if "device-side assert" in str(gen_e).lower() and self.use_gpu:
                        print("   🚨 CUDA Assert during generation. Switching to CPU...")
                        self.use_gpu = False
                        self.punjabi_model = self.punjabi_model.to('cpu')
                        inputs = {k: v.to('cpu') for k, v in inputs.items()}
                        generated_tokens = self.punjabi_model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=5,
                            num_return_sequences=1,
                            early_stopping=True
                        )
                    else:
                        raise gen_e
            
            # Decode
            translated_text = self.punjabi_tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True
            )
            
            return {
                'translated_text': translated_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'local_punjabi',
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ Local Punjabi translation failed: {e}")
            # Fallback to IndicBART
            return self._translate_with_indicbart(text, source_lang, target_lang)
    
    def _split_into_chunks(self, text: str, max_length: int = 4000) -> List[str]:
        """Split text into chunks for translation"""
        if len(text) <= max_length:
            return [text]
        
        # Split by sentences
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sent_len = len(sentence)
            
            if current_length + sent_len > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_len
            else:
                current_chunk.append(sentence)
                current_length += sent_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def translate_sections(
        self,
        sections: Dict[str, str],
        target_lang: str,
        auto_detect: bool = True
    ) -> Dict[str, Dict]:
        """
        Translate multiple sections
        
        Args:
            sections: Dict of section_name -> text
            target_lang: Target language code
            auto_detect: Auto-detect source language
            
        Returns:
            Dict of section_name -> translation_result
        """
        results = {}
        
        print(f"\n🌍 Translating sections to {self.LANGUAGE_NAMES.get(target_lang, target_lang)}...")
        
        for section_name, text in sections.items():
            print(f"\n  Translating: {section_name}")
            
            # Detect source language
            if auto_detect:
                detection = self.language_detector.detect_language(text)
                source_lang = detection['language']
                print(f"    Detected: {self.LANGUAGE_NAMES.get(source_lang, source_lang)}")
            else:
                source_lang = 'en'
            
            # Translate
            if source_lang == target_lang:
                print(f"    → Same language, skipping")
                results[section_name] = {
                    'translated_text': text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'method': 'none',
                    'success': True
                }
            else:
                result = self.translate(text, source_lang, target_lang)
                results[section_name] = result
                
                if result['success']:
                    orig_words = len(text.split())
                    trans_words = len(result['translated_text'].split())
                    print(f"    → Translated: {orig_words} → {trans_words} words")
                    print(f"    → Method: {result['method']}")
                else:
                    print(f"    → Translation failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def get_supported_languages(self) -> List[Tuple[str, str]]:
        """Get list of supported languages"""
        return [(code, name) for code, name in self.LANGUAGE_NAMES.items()]
    
    def detect_and_translate(
        self,
        text: str,
        target_lang: str
    ) -> Dict:
        """
        Auto-detect source language and translate
        """
        # Detect language
        detection = self.language_detector.detect_language(text)
        source_lang = detection['language']
        
        # Translate
        result = self.translate(text, source_lang, target_lang)
        result['detected_source'] = source_lang
        result['detection_confidence'] = detection['confidence']
        
        return result


# ============================================
# Convenience functions
# ============================================

def quick_translate(text: str, target_lang: str, source_lang: str = 'auto') -> str:
    """Quick translation utility"""
    translator = MultilingualTranslator()
    
    if source_lang == 'auto':
        result = translator.detect_and_translate(text, target_lang)
    else:
        result = translator.translate(text, source_lang, target_lang)
    
    return result['translated_text']


# ============================================
# Test function
# ============================================

def test_translation():
    """Test translation system"""
    translator = MultilingualTranslator(use_gpu=True)
    
    # Test samples
    tests = [
        {
            'text': 'The company achieved record revenue this year.',
            'source': 'en',
            'target': 'hi',
            'expected_contains': ['कंपनी', 'राजस्व']
        },
        {
            'text': 'कंपनी ने इस वर्ष रिकॉर्ड राजस्व हासिल किया।',
            'source': 'hi',
            'target': 'en',
            'expected_contains': ['company', 'revenue']
        }
    ]
    
    print("Testing translation system:\n")
    
    for i, test in enumerate(tests, 1):
        print(f"Test {i}:")
        print(f"  Original ({test['source']}): {test['text']}")
        
        result = translator.translate(
            test['text'],
            test['source'],
            test['target']
        )
        
        print(f"  Translated ({test['target']}): {result['translated_text']}")
        print(f"  Method: {result['method']}")
        print(f"  Success: {result['success']}")
        print(f"  Time: {result.get('time', 0):.2f}s")
        print()


if __name__ == "__main__":
    test_translation()