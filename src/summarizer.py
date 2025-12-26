# ============================================
# src/summarizer.py
# Unified summarization interface
# ============================================

from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
from rouge_score import rouge_scorer
import textstat

try:
    from .extractive_summarizer import ExtractiveSummarizer
    from .abstractive_summarizer import AbstractiveSummarizer, get_best_model
except ImportError as e:
    # If relative import fails, try absolute
    try:
        from extractive_summarizer import ExtractiveSummarizer
        from abstractive_summarizer import AbstractiveSummarizer, get_best_model
    except ImportError:
        # If both fail, raise the original error to see what's wrong (e.g. missing dependency inside the module)
        print(f"Import Error details: {e}")
        raise e


@dataclass
class SummaryResult:
    """Container for summary with metadata"""
    summary: str
    method: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: float
    quality_score: Optional[float] = None


class UnifiedSummarizer:
    """
    Unified interface for all summarization methods
    Provides easy access to extractive and abstractive approaches
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        abstractive_model: Optional[str] = None
    ):
        """
        Initialize unified summarizer
        
        Args:
            use_gpu: Whether to use GPU for abstractive models
            abstractive_model: Specific model name or None for auto-select
        """
        print("\n🚀 Initializing Unified Summarizer...")
        
        # Initialize extractive
        self.extractive = ExtractiveSummarizer()
        print("✓ Extractive summarizer ready")
        
        # Initialize abstractive (lazy loading)
        self.use_gpu = use_gpu
        if abstractive_model is None:
            abstractive_model = get_best_model(use_gpu)
        
        self.abstractive_model_name = abstractive_model
        self.abstractive = None  # Lazy load
        
        # Quality scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        print(f"✓ Ready! Abstractive model: {abstractive_model}")
    
    def _ensure_abstractive_loaded(self):
        """Load abstractive model if not already loaded"""
        if self.abstractive is None:
            self.abstractive = AbstractiveSummarizer(
                model_name=self.abstractive_model_name,
                device='cuda' if self.use_gpu else 'cpu'
            )
    
    def summarize(
        self,
        text: str,
        method: str = 'hybrid',
        ratio: float = 0.3,
        max_length: int = 150
    ) -> SummaryResult:
        """
        Generate summary using specified method
        
        Args:
            text: Input text
            method: 'extractive', 'abstractive', or 'hybrid'
            ratio: Compression ratio for extractive
            max_length: Max length for abstractive
            
        Returns:
            SummaryResult object
        """
        start_time = time.time()
        
        if method == 'extractive':
            summary = self.extractive.summarize(
                text,
                method='tfidf',
                ratio=ratio
            )
            used_method = 'extractive-tfidf'
            
        elif method == 'abstractive':
            self._ensure_abstractive_loaded()
            summary = self.abstractive.summarize(
                text,
                max_length=max_length
            )
            used_method = f'abstractive-{self.abstractive_model_name}'
            
        elif method == 'hybrid':
            # Use extractive first to reduce text, then abstractive
            if len(text.split()) > 500:
                # Long text: extract key content first
                extracted = self.extractive.summarize(
                    text,
                    method='bert',
                    ratio=0.5
                )
                self._ensure_abstractive_loaded()
                summary = self.abstractive.summarize(
                    extracted,
                    max_length=max_length
                )
                used_method = 'hybrid-extract-abstract'
            else:
                # Short text: use abstractive directly
                self._ensure_abstractive_loaded()
                summary = self.abstractive.summarize(
                    text,
                    max_length=max_length
                )
                used_method = 'abstractive-direct'
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        orig_words = len(text.split())
        summ_words = len(summary.split())
        compression = 1 - (summ_words / orig_words) if orig_words > 0 else 0
        
        return SummaryResult(
            summary=summary,
            method=used_method,
            original_length=orig_words,
            summary_length=summ_words,
            compression_ratio=compression,
            processing_time=processing_time
        )
    
    def summarize_sections(
        self,
        sections: Dict[str, str],
        method: str = 'hybrid',
        section_strategies: Optional[Dict[str, str]] = None
    ) -> Dict[str, SummaryResult]:
        """
        Summarize multiple sections intelligently
        
        Args:
            sections: Dict of section_name -> content
            method: Default method to use
            section_strategies: Override method for specific sections
        """
        results = {}
        
        print(f"\n📝 Summarizing {len(sections)} sections...")
        print(f"Default method: {method}")
        
        for section_name, content in sections.items():
            print(f"\n📄 {section_name}:")
            
            # Get method for this section
            if section_strategies and section_name in section_strategies:
                section_method = section_strategies[section_name]
            else:
                section_method = method
            
            # Determine parameters based on section type
            params = self._get_section_params(section_name, content)
            
            # Summarize
            result = self.summarize(
                content,
                method=section_method,
                ratio=params['ratio'],
                max_length=params['max_length']
            )
            
            results[section_name] = result
            
            # Print statistics
            print(f"  Method: {result.method}")
            print(f"  {result.original_length} → {result.summary_length} words "
                  f"({result.compression_ratio*100:.1f}% compression)")
            print(f"  Time: {result.processing_time:.2f}s")
        
        return results
    
    def _get_section_params(self, section_name: str, content: str) -> Dict:
        """Get summarization parameters for specific section"""
        word_count = len(content.split())
        
        # Section-specific strategies
        strategies = {
            "Chairman's Message": {
                'ratio': 0.35,
                'max_length': 130
            },
            "Executive Summary": {
                'ratio': 0.4,
                'max_length': 150
            },
            "Financial Performance": {
                'ratio': 0.25,
                'max_length': 200
            },
            "Management Discussion": {
                'ratio': 0.3,
                'max_length': 250
            },
            "Risk Management": {
                'ratio': 0.35,
                'max_length': 150
            },
            "Future Outlook": {
                'ratio': 0.4,
                'max_length': 120
            }
        }
        
        # Get strategy or use default
        if section_name in strategies:
            return strategies[section_name]
        else:
            return {
                'ratio': 0.3,
                'max_length': 150
            }
    
    def evaluate_summary(
        self,
        original: str,
        summary: str,
        reference: Optional[str] = None
    ) -> Dict:
        """
        Evaluate summary quality
        
        Args:
            original: Original text
            summary: Generated summary
            reference: Reference summary (optional)
        
        Returns:
            Dict of quality metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['original_words'] = len(original.split())
        metrics['summary_words'] = len(summary.split())
        metrics['compression_ratio'] = 1 - (
            metrics['summary_words'] / metrics['original_words']
        )
        
        # Readability
        metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(summary)
        metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(summary)
        
        # ROUGE scores (if reference provided)
        if reference:
            rouge_scores = self.rouge_scorer.score(reference, summary)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # Information retention (simple metric)
        # Check if key entities are preserved
        from collections import Counter
        import re
        
        # Extract numbers and capitalized words (likely important)
        orig_numbers = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', original))
        summ_numbers = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', summary))
        
        orig_caps = set(re.findall(r'\b[A-Z][a-z]+\b', original))
        summ_caps = set(re.findall(r'\b[A-Z][a-z]+\b', summary))
        
        if orig_numbers:
            metrics['number_retention'] = len(summ_numbers & orig_numbers) / len(orig_numbers)
        else:
            metrics['number_retention'] = 1.0
        
        if orig_caps:
            metrics['entity_retention'] = len(summ_caps & orig_caps) / len(orig_caps)
        else:
            metrics['entity_retention'] = 1.0
        
        return metrics
    
    def get_best_summary(
        self,
        text: str,
        methods: List[str] = ['extractive', 'abstractive', 'hybrid']
    ) -> Tuple[SummaryResult, Dict[str, SummaryResult]]:
        """
        Generate summaries with multiple methods and pick the best
        
        Returns:
            (best_result, all_results)
        """
        results = {}
        
        print("\n🔬 Generating summaries with multiple methods...")
        
        for method in methods:
            print(f"\n  Trying {method}...")
            result = self.summarize(text, method=method)
            results[method] = result
        
        # Evaluate and pick best
        best_method = None
        best_score = -1
        
        for method, result in results.items():
            # Simple scoring: balance compression and readability
            metrics = self.evaluate_summary(text, result.summary)
            
            # Score = readability * (1 - abs(compression - 0.7))
            # Prefer ~70% compression with good readability
            score = (
                (metrics['flesch_reading_ease'] / 100) *
                (1 - abs(result.compression_ratio - 0.7))
            )
            
            if score > best_score:
                best_score = score
                best_method = method
        
        print(f"\n✓ Best method: {best_method} (score: {best_score:.3f})")
        
        return results[best_method], results


# ============================================
# Convenience function
# ============================================

def quick_summarize(
    text: str,
    method: str = 'hybrid',
    use_gpu: bool = True
) -> str:
    """
    Quick one-liner summarization
    """
    summarizer = UnifiedSummarizer(use_gpu=use_gpu)
    result = summarizer.summarize(text, method=method)
    return result.summary