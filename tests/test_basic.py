import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.summarizer import Summarizer
from src.multilingual import MultilingualHandler

class TestAnnualReportSummarizer(unittest.TestCase):
    
    def setUp(self):
        self.sample_pdf = "data/sample_reports/AR_26481_INFY_2024_2025_A_02062025153945.pdf"
        self.processor = PDFProcessor(self.sample_pdf)
        self.summarizer = Summarizer()
        self.multilingual = MultilingualHandler()

    def test_pdf_extraction(self):
        """Test if text can be extracted from PDF"""
        text = self.processor.extract_text()
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

    def test_section_identification(self):
        """Test if sections are identified"""
        self.processor.extract_text()
        sections = self.processor.identify_sections()
        self.assertIsInstance(sections, dict)
        # We expect at least some sections to be found in a real annual report
        self.assertGreater(len(sections), 0)

    def test_summarization_extractive(self):
        """Test extractive summarization"""
        text = "This is a test sentence. This is another test sentence. And a third one."
        summary = self.summarizer.extractive_summary(text, sentences_count=1)
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 0)

    def test_language_detection(self):
        """Test language detection"""
        text = "This is English"
        lang = self.multilingual.detect_language(text)
        self.assertEqual(lang, 'en')
        
        text_hi = "यह हिंदी है"
        lang_hi = self.multilingual.detect_language(text_hi)
        self.assertEqual(lang_hi, 'hi')

if __name__ == '__main__':
    unittest.main()
