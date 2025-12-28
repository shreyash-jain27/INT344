# ============================================
# src/multilingual_summarizer.py
# Summarization for multiple Indian languages
# ============================================

from typing import Dict, List, Optional
import time

try:
    from .language_detector import LanguageDetector
    from .translator import MultilingualTranslator
    from .multilingual_processor import MultilingualProcessor
    from .summarizer import UnifiedSummarizer, SummaryResult
except:
    from language_detector import LanguageDetector
    from translator import MultilingualTranslator
    from multilingual_processor import MultilingualProcessor
    from summarizer import UnifiedSummarizer, SummaryResult


class MultilingualSummarizer:
    """
    Summarization system for multiple Indian languages
    Supports direct summarization and translate-summarize-translate approach
    """
    
    def __init__(self, use_gpu: bool = True):
        import torch
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.translator = MultilingualTranslator(use_gpu=use_gpu)
        self.text_processor = MultilingualProcessor()
        
        # English summarizer (already built)
        self.english_summarizer = UnifiedSummarizer(use_gpu=use_gpu)
        
        # IndicBART for direct Indian language summarization
        self.indic_model = None
        self.indic_tokenizer = None
        
        # mBART for multilingual (fallback)
        self.mbart_pipeline = None
        
        print("🌍 Multilingual Summarizer initialized")
    
    def _load_indicbart(self):
        """Load IndicBART model for summarization"""
        if self.indic_model is not None:
            return
        
        print("\n🤖 Loading IndicBART for summarization...")
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_name = "ai4bharat/IndicBARTSS"
            
            self.indic_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                # use_fast=False
            )
            
            self.indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_safetensors=True
            )
            
            if self.use_gpu:
                self.indic_model = self.indic_model.to('cuda')
                self.indic_model = self.indic_model.half()  # FP16 for speed
            
            print("   ✓ IndicBART loaded!")
            
        except Exception as e:
            print(f"   ⚠️ Could not load IndicBART: {e}")
    
    def _load_mbart(self):
        """Load mBART as fallback"""
        if self.mbart_pipeline is not None:
            return
        
        print("\n🤖 Loading mBART...")
        
        try:
            from transformers import pipeline
            self.mbart_pipeline = pipeline(
                "summarization",
                model="facebook/mbart-large-50",
                device=0 if self.use_gpu else -1,
                model_kwargs={"use_safetensors": True}
            )
            print("   ✓ mBART loaded!")
        except Exception as e:
            print(f"   ⚠️ Could not load mBART: {e}")
    
    def summarize(
        self,
        text: str,
        source_lang: Optional[str] = None,
        output_lang: Optional[str] = None,
        method: str = 'auto',
        max_length: int = 150,
        compression: Optional[float] = None
    ) -> Dict:
        """
        Summarize text in any supported language
        
        Args:
            text: Input text
            source_lang: Source language (auto-detect if None)
            output_lang: Output language (same as source if None)
            method: 'direct', 'translate', or 'auto'
            max_length: Maximum summary length
            
        Returns:
            Dict with summary and metadata
        """
        start_time = time.time()
        
        # Detect source language if not provided
        if source_lang is None:
            detection = self.language_detector.detect_language(text)
            source_lang = detection['language']
            confidence = detection['confidence']
            print(f"  Detected language: {source_lang} ({confidence*100:.1f}%)")
        
        # Default output language to source
        if output_lang is None:
            output_lang = source_lang
        
        # Choose method
        if method == 'auto':
            # Use direct only if source and output are the same (especially English)
            if source_lang == output_lang:
                method = 'direct'
            # Otherwise use translate-summarize-translate for cross-lingual needs
            else:
                method = 'translate'
        
        # Determine max_length from compression if provided
        if compression is not None:
            words = text.split()
            word_count = len(words)
            
            # For very large documents, we need sensible bounds
            # Floor of 100 words or 5%, ceiling of 800 words
            target = int(word_count * compression)
            max_length = max(100, min(800, target))
            
            # If the original is shorter than 150 words, don't force it to be 100
            if word_count < 150:
                max_length = max(30, target)
                
            print(f"  Target length for {source_lang}: {max_length} words (original: {word_count}, choice: {compression*100:.0f}%)")

        # Preliminary text cleaning
        text = self._clean_input_text(text)

        # Perform summarization
        if method == 'direct':
            if source_lang == 'en':
                # Use English summarizer
                result = self.english_summarizer.summarize(
                    text,
                    method='hybrid',
                    max_length=max_length
                )
                summary = result.summary
                summ_method = 'english-hybrid'
            else:
                # Try IndicBART
                summary = self._summarize_with_indicbart(
                    text, source_lang, max_length
                )
                summ_method = 'indicbart'
        
        elif method == 'translate':
            # Translate to English, summarize, translate back
            summary = self._translate_summarize_translate(
                text, source_lang, output_lang, max_length
            )
            summ_method = 'translate-summarize-translate'
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate statistics
        orig_words = len(text.split())
        summ_words = len(summary.split())
        compression = 1 - (summ_words / orig_words) if orig_words > 0 else 0
        
        return {
            'summary': summary,
            'source_lang': source_lang,
            'output_lang': output_lang,
            'method': summ_method,
            'original_length': orig_words,
            'summary_length': summ_words,
            'compression_ratio': compression,
            'processing_time': time.time() - start_time
        }
    
    def _summarize_with_indicbart(
        self,
        text: str,
        lang: str,
        max_length: int
    ) -> str:
        """Direct summarization with IndicBART"""
        try:
            self._load_indicbart()
            
            if self.indic_model is None:
                # Fallback
                return self._translate_summarize_translate(text, lang, lang, max_length)
            
            # Prepare input
            input_text = f"<{lang}> {text}"
            
            inputs = self.indic_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            )
            
            if self.use_gpu:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate summary
            import torch
            with torch.no_grad():
                summary_ids = self.indic_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=30,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            summary = self.indic_tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            return summary
            
        except Exception as e:
            print(f"⚠️ IndicBART summarization failed: {e}")
            # Fallback
            return self._translate_summarize_translate(text, lang, lang, max_length)
    
    def _translate_summarize_translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int
    ) -> str:
        """
        Three-step approach:
        1. Translate to English
        2. Summarize in English
        3. Translate summary back
        """
        print(f"  Using translate-summarize-translate method")
        
        # Step 1: Translate to English if needed
        if source_lang != 'en':
            print(f"  Step 1: Translating {source_lang} → en")
            trans_result = self.translator.translate(text, source_lang, 'en')
            english_text = trans_result['translated_text']
        else:
            english_text = text
        
        # Step 2: Summarize in English
        print(f"  Step 2: Summarizing in English")
        summ_result = self.english_summarizer.summarize(
            english_text,
            method='hybrid',
            max_length=max_length
        )
        english_summary = summ_result.summary
        
        # Step 3: Translate summary back if needed
        if target_lang != 'en':
            print(f"  Step 3: Translating en → {target_lang}")
            back_trans = self.translator.translate(
                english_summary, 'en', target_lang
            )
            final_summary = back_trans['translated_text']
        else:
            final_summary = english_summary
        
        return final_summary
    
    def summarize_sections(
        self,
        sections: Dict[str, str],
        output_lang: Optional[str] = None,
        auto_detect: bool = True,
        method: str = 'auto',
        compression: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        Summarize multiple sections with multilingual support
        
        Args:
            sections: Dict of section_name -> text
            output_lang: Desired output language (None = keep original)
            auto_detect: Auto-detect language per section
            method: Summarization method
            
        Returns:
            Dict of section_name -> summary_result
        """
        results = {}
        
        print(f"\n🌍 Multilingual Section Summarization")
        if output_lang:
            print(f"Output language: {self.translator.LANGUAGE_NAMES.get(output_lang, output_lang)}")
        
        for section_name, text in sections.items():
            print(f"\n📄 {section_name}:")
            
            # Skip very short sections
            if len(text.split()) < 50:
                results[section_name] = {
                    'summary': text,
                    'source_lang': 'unknown',
                    'output_lang': output_lang or 'unknown',
                    'method': 'too_short',
                    'original_length': len(text.split()),
                    'summary_length': len(text.split()),
                    'compression_ratio': 0.0,
                    'processing_time': 0.0
                }
                print(f"  → Too short, keeping original")
                continue
            
            # Summarize
            result = self.summarize(
                text,
                source_lang=None if auto_detect else 'en',
                output_lang=output_lang,
                method=method,
                compression=compression
            )
            
            results[section_name] = result
            
            # Print statistics
            print(f"  → {result['original_length']} → {result['summary_length']} words")
            print(f"  → Method: {result['method']}")
            print(f"  → Time: {result['processing_time']:.2f}s")
        
        return results
    
    def get_summary_in_multiple_languages(
        self,
        text: str,
        target_languages: List[str],
        source_lang: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Generate summaries in multiple languages
        
        Args:
            text: Input text
            target_languages: List of target language codes
            source_lang: Source language (auto-detect if None)
            
        Returns:
            Dict mapping language code to summary result
        """
        results = {}
        
        # Detect source if needed
        if source_lang is None:
            detection = self.language_detector.detect_language(text)
            source_lang = detection['language']
        
        print(f"\n🌍 Generating summaries in {len(target_languages)} languages...")
        
        for lang in target_languages:
            print(f"\n  Language: {self.translator.LANGUAGE_NAMES.get(lang, lang)}")
            result = self.summarize(
                text,
                source_lang=source_lang,
                output_lang=lang,
                method='auto'
            )
            results[lang] = result
        
        return results

    def _clean_input_text(self, text: str) -> str:
        """Basic cleaning for input text to avoid model artifacts"""
        if not text:
            return ""
        
        # Remove weird artifacts like "@ info/ plain"
        import re
        text = re.sub(r'@[^ ]+/[^ ]+', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()


# ============================================
# Convenience functions
# ============================================

def quick_multilingual_summary(
    text: str,
    output_lang: str = 'en'
) -> str:
    """Quick multilingual summarization"""
    summarizer = MultilingualSummarizer(use_gpu=True)
    result = summarizer.summarize(text, output_lang=output_lang)
    return result['summary']


# ============================================
# Test function
# ============================================

def test_multilingual_summarization():
    """Test multilingual summarization"""
    summarizer = MultilingualSummarizer(use_gpu=True)
    
    # Test cases
    tests = [
        {
            'name': 'English to English',
            'text': '''The company achieved record revenue of $10 billion in fiscal year 2023,
                      representing a 25% increase from the previous year. This growth was driven
                      by strong performance across all business segments. Technology division
                      revenue grew 40% year-over-year. The company continues to invest heavily
                      in research and development.''',
            'source_lang': 'en',
            'output_lang': 'en'
        },
        {
            'name': 'English to Hindi',
            'text': '''The company achieved record revenue of $10 billion in fiscal year 2023,
                      representing a 25% increase from the previous year.''',
            'source_lang': 'en',
            'output_lang': 'hi'
        }
    ]
    
    print("Testing multilingual summarization:\n")
    
    for test in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print('='*60)
        print(f"Original ({test['source_lang']}): {test['text'][:100]}...")
        
        result = summarizer.summarize(
            test['text'],
            source_lang=test['source_lang'],
            output_lang=test['output_lang']
        )
        
        print(f"\nSummary ({test['output_lang']}): {result['summary']}")
        print(f"Method: {result['method']}")
        print(f"Compression: {result['compression_ratio']*100:.1f}%")
        print(f"Time: {result['processing_time']:.2f}s")



if __name__ == "__main__":
    test_multilingual_summarization()