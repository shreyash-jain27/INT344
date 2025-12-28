# ============================================
# src/extractive_summarizer.py
# Extractive summarization using Sumy and BERT
# ============================================

import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import numpy as np
from typing import List, Optional, Union

# Ensure nltk resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ExtractiveSummarizer:
    """
    Extractive summarization using statistical (Sumy) and embedding (BERT) methods
    """
    
    def __init__(self):
        self.stemmer = Stemmer("english")
        self.stop_words = get_stop_words("english")
        self.bert_model = None
        self._sentence_transformer = None
        self._kmeans = None
        self._pairwise_distances_argmin_min = None

    def summarize(
        self, 
        text: str, 
        method: str = 'lsa', 
        ratio: float = 0.2, 
        sentences_count: Optional[int] = None
    ) -> str:
        """
        Generate extractive summary
        
        Args:
            text: Input text
            method: 'lsa', 'lexrank', 'textrank', 'tfidf', or 'bert'
            ratio: Compression ratio (0.0 to 1.0)
            sentences_count: Exact number of sentences (overrides ratio)
            
        Returns:
            Summary string
        """
        if not text or not text.strip():
            return ""
            
        # Handle 'tfidf' as LSA for compatibility
        if method == 'tfidf':
            method = 'lsa'
            
        # Calculate sentence count
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            total_sentences = len(parser.document.sentences)
            
            if total_sentences == 0:
                return ""
                
            if sentences_count is None:
                sentences_count = max(1, int(total_sentences * ratio))
                
            if sentences_count >= total_sentences:
                return text
                
        except Exception as e:
            print(f"Error parsing text: {e}")
            return text

        # BERT Method
        if method == 'bert':
            return self._summarize_bert(text, sentences_count)
        
        # Sumy Methods
        try:
            if method == 'lexrank':
                summarizer = LexRankSummarizer(self.stemmer)
            elif method == 'textrank':
                summarizer = TextRankSummarizer(self.stemmer)
            else: # Default to LSA
                summarizer = LsaSummarizer(self.stemmer)
                
            summarizer.stop_words = self.stop_words
            
            summary_sentences = summarizer(parser.document, sentences_count)
            return " ".join([str(s) for s in summary_sentences])
            
        except Exception as e:
            print(f"Error in extractive summarization ({method}): {e}")
            # Fallback to first N sentences
            sentences = nltk.sent_tokenize(text)
            return " ".join(sentences[:sentences_count])

    def _summarize_bert(self, text: str, num_sentences: int) -> str:
        """
        BERT-based extractive summarization using K-Means clustering of sentence embeddings
        """
        # Lazy load dependencies
        if self.bert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.cluster import KMeans
                from sklearn.metrics import pairwise_distances_argmin_min
                
                print("Loading BERT model for extractive summarization...")
                # Robust loading with CUDA error handling
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                try:
                    self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    print(f"✓ BERT model loaded on {device}")
                except Exception as gpu_err:
                    if device == "cuda":
                        print(f"⚠️ GPU load failed, falling back to CPU: {gpu_err}")
                        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
                        print("✓ BERT model loaded on CPU")
                    else:
                        raise gpu_err
                
                self._kmeans = KMeans
                self._pairwise_distances_argmin_min = pairwise_distances_argmin_min
            except ImportError:
                print("⚠️ sentence-transformers or sklearn not installed. Falling back to LSA.")
                return self.summarize(text, method='lsa', sentences_count=num_sentences)
            
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
            
        try:
            # Embed sentences
            try:
                embeddings = self.bert_model.encode(sentences)
            except Exception as e:
                if "CUDA error" in str(e) or "device-side assert" in str(e):
                    print(f"⚠️ CUDA error detected during BERT encoding: {e}. Switching model to CPU...")
                    # This is tricky because SentenceTransformer might keep the old device state
                    # We'll try to re-init on CPU
                    from sentence_transformers import SentenceTransformer
                    self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                    embeddings = self.bert_model.encode(sentences)
                else:
                    raise e
            
            # Cluster
            num_clusters = min(num_sentences, len(sentences))
            kmeans = self._kmeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            
            # Find closest sentence to each cluster center
            closest, _ = self._pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
            closest = sorted(closest)
            
            return " ".join([sentences[i] for i in closest])
            
        except Exception as e:
            print(f"⚠️ BERT summarization failed completely: {e}. Falling back to LSA.")
            return self.summarize(text, method='lsa', sentences_count=num_sentences)

if __name__ == "__main__":
    # Test
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Tesla), automated decision-making and competing at the highest level in strategic game systems (such as chess and Go).
    """
    
    summ = ExtractiveSummarizer()
    print("LSA Summary:", summ.summarize(text, method='lsa', ratio=0.5))
    print("\nBERT Summary:", summ.summarize(text, method='bert', ratio=0.5))