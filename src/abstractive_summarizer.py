# ============================================
# src/abstractive_summarizer.py
# Transformer-based abstractive summarization
# ============================================

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BartTokenizer
)
from typing import List, Dict, Optional
import re
import numpy as np
from nltk.tokenize import sent_tokenize


class AbstractiveSummarizer:
    """
    Abstractive summarization using transformer models:
    - BART (facebook/bart-large-cnn)
    - T5 (t5-base/t5-small)
    - LED (for long documents)
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/bart-large-cnn',
        device: Optional[str] = None
    ):
        """
        Initialize abstractive summarizer
        
        Args:
            model_name: Hugging Face model name
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1
        
        print(f"\n🤖 Initializing {model_name}...")
        print(f"   Device: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")
        
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Lazy loading - only load when first used
        self._is_loaded = False
    
    def _load_model(self):
        """Load model (lazy loading)"""
        if self._is_loaded:
            return
        
        print("   Loading model (this may take a minute)...")
        
        try:
            # Try pipeline first (easier)
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device == 0 else torch.float32
            )
            
            print("   ✓ Model loaded successfully!")
            
            # Always load tokenizer for accurate chunking
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._is_loaded = True
            
        except Exception as e:
            print(f"   ⚠️ Pipeline failed, trying manual loading: {e}")
            
            # Manual loading
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 0 else torch.float32
                )
                
                if self.device == 0:
                    self.model = self.model.to('cuda')
                
                print("   ✓ Model loaded successfully!")
                self._is_loaded = True
                
            except Exception as e2:
                print(f"   ❌ Failed to load model: {e2}")
                raise
    
    def _chunk_text(self, text: str, max_length: int = 1024) -> List[str]:
        """
        Split long text into chunks for processing
        """
        # Tokenize to check length
        if self.tokenizer is None:
            # Approximate chunking (Conservative: 1 token ≈ 0.75 words)
            # BART limit is 1024 tokens. 
            # Safe word limit ≈ 1024 * 0.75 ≈ 750 words
            words = text.split()
            chunk_size = int(max_length * 0.6)  # Very conservative to be safe
            
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
            return chunks
        
        # Proper tokenization
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_length:
            return [text]
        
        # Split by sentences to avoid breaking mid-sentence
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sent_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sent_length = len(sent_tokens)
            
            if current_length + sent_length > max_length:
                # Start new chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_length
            else:
                current_chunk.append(sentence)
                current_length += sent_length
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
        do_sample: bool = False,
        length_penalty: float = 2.0,
        num_beams: int = 4
    ) -> str:
        """
        Generate abstractive summary
        
        Args:
            text: Input text
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            do_sample: Whether to use sampling
            length_penalty: Length penalty for beam search
            num_beams: Number of beams for beam search
            
        Returns:
            Summary text
        """
        if not text or len(text.strip()) < 50:
            return text
        
        # Load model if not already loaded
        self._load_model()
        
        # Chunk if text is too long
        chunks = self._chunk_text(text, max_length=1024)
        num_chunks = len(chunks)
        
        # If we have multiple chunks, max_length per chunk should be smaller
        # so that the combined result matches the target max_length
        if num_chunks > 1:
            chunk_max_length = max(min_length + 20, int(max_length / num_chunks))
            # But don't go below min_length
            chunk_max_length = max(chunk_max_length, 80) 
            print(f"    Dividing work into {num_chunks} chunks. Target per chunk: ~{chunk_max_length} words")
        else:
            chunk_max_length = max_length

        summaries = []
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"    Processing chunk {i+1}/{len(chunks)}...")
            
            try:
                if self.pipeline:
                    # Use pipeline
                    result = self.pipeline(
                        chunk,
                        max_length=chunk_max_length,
                        min_length=min(min_length, chunk_max_length - 10),
                        do_sample=do_sample,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        early_stopping=True,
                        truncation=True  # CRITICAL: Prevent index out of bounds
                    )
                    summary = result[0]['summary_text']
                    
                else:
                    # Use model directly
                    inputs = self.tokenizer(
                        chunk,
                        max_length=1024,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    if self.device == 0:
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    
                    summary_ids = self.model.generate(
                        inputs['input_ids'],
                        max_length=chunk_max_length,
                        min_length=min(min_length, chunk_max_length - 10),
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                    
                    summary = self.tokenizer.decode(
                        summary_ids[0],
                        skip_special_tokens=True
                    )
                
                summaries.append(summary)
                
            except Exception as e:
                print(f"    ⚠️ Error processing chunk: {e}")
                # Use first 150 words as fallback
                summaries.append(' '.join(chunk.split()[:150]) + '...')
        
        # Combine summaries
        final_summary = ' '.join(summaries)
        
        return final_summary
    
    def summarize_with_context(
        self,
        text: str,
        context: str = "",
        max_length: int = 150
    ) -> str:
        """
        Summarize with additional context
        Useful for section-aware summarization
        """
        if context:
            # Prepend context to help the model
            input_text = f"{context}: {text}"
        else:
            input_text = text
        
        return self.summarize(input_text, max_length=max_length)
    
    def summarize_sections(
        self,
        sections: Dict[str, str],
        max_length: int = 150,
        section_strategies: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, str]:
        """
        Summarize multiple sections with section-specific strategies
        
        Args:
            sections: Dictionary of section_name -> content
            max_length: Default max length
            section_strategies: Section-specific parameters
        """
        summaries = {}
        
        print(f"\n📝 Generating abstractive summaries ({self.model_name})...")
        
        # Default strategies for different sections
        default_strategies = {
            "Chairman's Message": {
                'max_length': 130,
                'context': 'Leadership message'
            },
            "Financial Performance": {
                'max_length': 180,
                'context': 'Financial highlights'
            },
            "Risk Management": {
                'max_length': 150,
                'context': 'Key risks'
            },
            "Future Outlook": {
                'max_length': 140,
                'context': 'Future plans'
            }
        }
        
        # Merge with user strategies
        if section_strategies:
            strategies = {**default_strategies, **section_strategies}
        else:
            strategies = default_strategies
        
        for section_name, content in sections.items():
            print(f"\n  Section: {section_name}")
            
            if len(content.strip()) < 100:
                summaries[section_name] = content
                print(f"    → Too short, keeping original")
                continue
            
            # Get strategy for this section
            strategy = strategies.get(section_name, {})
            section_max_length = strategy.get('max_length', max_length)
            context = strategy.get('context', '')
            
            # Generate summary
            summary = self.summarize_with_context(
                content,
                context=context,
                max_length=section_max_length
            )
            
            summaries[section_name] = summary
            
            # Statistics
            orig_words = len(content.split())
            summ_words = len(summary.split())
            compression = (1 - summ_words/orig_words) * 100
            
            print(f"    → {orig_words} words → {summ_words} words ({compression:.1f}% compression)")
        
        return summaries
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'model_name': self.model_name,
            'device': 'GPU (CUDA)' if self.device == 0 else 'CPU',
            'is_loaded': self._is_loaded,
            'using_pipeline': self.pipeline is not None
        }


# ============================================
# Utility function for best model selection
# ============================================

def get_best_model(use_gpu: bool = True) -> str:
    """
    Select best model based on available resources
    """
    if not use_gpu:
        return 't5-small'  # Smaller, faster on CPU
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory >= 6:  # 6GB or more
            return 'facebook/bart-large-cnn'  # Best quality
        elif gpu_memory >= 4:
            return 't5-base'  # Good balance
        else:
            return 't5-small'  # Smaller footprint
    
    return 't5-small'

