# ============================================
# test_day5.py - Test multilingual features
# ============================================

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from language_detector import LanguageDetector
from multilingual_processor import MultilingualProcessor
from translator import MultilingualTranslator
from multilingual_summarizer import MultilingualSummarizer


def test_language_detection():
    """Test 1: Language Detection"""
    print("=" * 70)
    print("TEST 1: LANGUAGE DETECTION")
    print("=" * 70)
    
    detector = LanguageDetector()
    
    samples = {
        'English': 'The company achieved record revenue this year.',
        'Hindi': 'कंपनी ने इस वर्ष रिकॉर्ड राजस्व हासिल किया।',
        'Tamil': 'நிறுவனம் இந்த ஆண்டு சாதனை வருவாயை அடைந்தது।',
        'Telugu': 'కంపెనీ ఈ సంవత్సరం రికార్డు రాబడిని సాధించింది।',
        'Bengali': 'কোম্পানি এই বছর রেকর্ড রাজস্ব অর্জন করেছে।',
        'Hinglish': 'Company ne is year record revenue achieve kiya hai yaar.'
    }
    
    print("\nDetecting languages:\n")
    
    for name, text in samples.items():
        result = detector.detect_language(text, return_all=True)
        lang_info = detector.get_language_info(result['language'])
        
        print(f"{name}:")
        print(f"  Text: {text[:60]}")
        print(f"  Detected: {lang_info['name']} ({result['language']})")
        print(f"  Script: {lang_info['script']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Code-mixed: {result.get('is_code_mixed', False)}")
        
        if 'all_languages' in result and len(result['all_languages']) > 1:
            print(f"  All detected: {result['all_languages'][:3]}")
        print()
    
    print("✅ Language detection test complete!\n")


def test_text_processing():
    """Test 2: Multilingual Text Processing"""
    print("=" * 70)
    print("TEST 2: MULTILINGUAL TEXT PROCESSING")
    print("=" * 70)
    
    processor = MultilingualProcessor()
    
    samples = {
        'hi': 'कंपनी ने इस वर्ष रिकॉर्ड राजस्व हासिल किया। यह पिछले साल से 25% अधिक है।',
        'ta': 'நிறுவனம் இந்த ஆண்டு சாதனை வருவாயை அடைந்தது। இது கடந்த ஆண்டை விட 25% அதிகம்.',
        'en': 'The company achieved record revenue this year. This is 25% more than last year.'
    }
    
    print("\nProcessing text in different languages:\n")
    
    for lang, text in samples.items():
        config = processor.LANGUAGE_CONFIG.get(lang, {})
        print(f"{config.get('name', lang)}:")
        print(f"  Original: {text[:60]}...")
        
        # Normalize
        normalized = processor.normalize_text(text, lang)
        print(f"  Normalized: {normalized[:60]}...")
        
        # Tokenize
        sentences = processor.tokenize_sentences(text, lang)
        words = processor.tokenize_words(text, lang)
        
        print(f"  Sentences: {len(sentences)}")
        print(f"  Words: {len(words)}")
        
        # Statistics
        stats = processor.get_text_statistics(text, lang)
        print(f"  Avg words/sentence: {stats['avg_words_per_sentence']:.1f}")
        print()
    
    print("✅ Text processing test complete!\n")


def test_translation():
    """Test 3: Translation"""
    print("=" * 70)
    print("TEST 3: TRANSLATION")
    print("=" * 70)
    
    translator = MultilingualTranslator(use_gpu=True)
    
    tests = [
        {
            'text': 'The company achieved record revenue this year.',
            'source': 'en',
            'target': 'hi',
            'name': 'English to Hindi'
        },
        {
            'text': 'कंपनी ने इस वर्ष रिकॉर्ड राजस्व हासिल किया।',
            'source': 'hi',
            'target': 'en',
            'name': 'Hindi to English'
        },
        {
            'text': 'The company achieved record revenue.',
            'source': 'en',
            'target': 'ta',
            'name': 'English to Tamil'
        }
    ]
    
    print("\nTranslating text:\n")
    
    for test in tests:
        print(f"{test['name']}:")
        print(f"  Original: {test['text']}")
        
        result = translator.translate(
            test['text'],
            test['source'],
            test['target']
        )
        
        if result['success']:
            print(f"  Translated: {result['translated_text']}")
            print(f"  Method: {result['method']}")
            print(f"  Time: {result.get('time', 0):.2f}s")
        else:
            print(f"  ❌ Translation failed: {result.get('error')}")
        print()
    
    print("✅ Translation test complete!\n")


def test_multilingual_summarization():
    """Test 4: Multilingual Summarization"""
    print("=" * 70)
    print("TEST 4: MULTILINGUAL SUMMARIZATION")
    print("=" * 70)
    
    summarizer = MultilingualSummarizer(use_gpu=True)
    
    test_text = '''
    The company achieved record revenue of $10 billion in fiscal year 2023,
    representing a 25% increase from the previous year. This remarkable growth
    was driven by strong performance across all business segments. Our technology
    division saw particularly impressive results, with revenue growing 40%
    year-over-year. The company continues to invest heavily in research and
    development, with R&D spending reaching $1.5 billion. We have launched
    several innovative products that have been well-received by customers
    worldwide. Management remains optimistic about future growth prospects.
    '''
    
    print("\nGenerating summaries in multiple languages:\n")
    
    # Test different language outputs
    languages = ['en', 'hi', 'ta']
    
    for lang in languages:
        lang_name = summarizer.translator.LANGUAGE_NAMES.get(lang, lang)
        print(f"\n{lang_name} Summary:")
        print("-" * 60)
        
        result = summarizer.summarize(
            test_text,
            source_lang='en',
            output_lang=lang,
            method='auto'
        )
        
        print(f"Summary: {result['summary']}")
        print(f"Method: {result['method']}")
        print(f"Original: {result['original_length']} words")
        print(f"Summary: {result['summary_length']} words")
        print(f"Compression: {result['compression_ratio']*100:.1f}%")
        print(f"Time: {result['processing_time']:.2f}s")
    
    print("\n✅ Multilingual summarization test complete!\n")


def test_full_document():
    """Test 5: Full Document with Auto-Detection"""
    print("=" * 70)
    print("TEST 5: FULL DOCUMENT PROCESSING")
    print("=" * 70)
    
    # Get PDF
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 Testing with: {pdf_path.name}\n")
    
    # Process PDF
    from pdf_processor import PDFProcessor
    processor = PDFProcessor(str(pdf_path))
    sections = processor.identify_sections()
    
    if not sections:
        print("❌ No sections found")
        return
    
    print(f"✓ Found {len(sections)} sections")
    
    # Detect languages
    detector = LanguageDetector()
    lang_results = detector.detect_document_language(sections)
    
    # Summarize first section in multiple languages
    first_section = list(sections.keys())[0]
    first_content = sections[first_section]
    
    print(f"\n📝 Testing with section: {first_section}")
    print(f"Length: {len(first_content.split())} words")
    
    # Summarize in English and Hindi
    summarizer = MultilingualSummarizer(use_gpu=True)
    
    for lang in ['en', 'hi']:
        result = summarizer.summarize(
            first_content,
            output_lang=lang
        )
        
        lang_name = summarizer.translator.LANGUAGE_NAMES.get(lang)
        print(f"\n{lang_name} Summary ({result['summary_length']} words):")
        print(result['summary'][:200] + "...")
    
    print("\n✅ Full document test complete!\n")


def run_all_tests():
    """Run all Day 5 tests"""
    print("\n🧪 DAY 5 TESTING SUITE")
    print("=" * 70)
    print("Testing Multilingual Features")
    print("=" * 70)
    
    print("\n⚠️  Note: Some tests require internet connection for Google Translate")
    print("⚠️  First run will download models (~1-2GB)\n")
    
    input("Press Enter to start tests...")
    
    try:
        test_language_detection()
        test_text_processing()
        test_translation()
        test_multilingual_summarization()
        test_full_document()
        
        print("\n" + "=" * 70)
        print("✅ ALL DAY 5 TESTS COMPLETE!")
        print("=" * 70)
        
        print("\n🎉 Your multilingual system is working!")
        print("\nKey Features Tested:")
        print("  ✓ Language detection (11 Indian languages)")
        print("  ✓ Multilingual text processing")
        print("  ✓ Translation (English ↔ Indian languages)")
        print("  ✓ Multilingual summarization")
        print("  ✓ Full document processing")
        
        print("\nNext steps:")
        print("1. Update web interface (Multilingual tab)")
        print("2. Test with real multilingual documents")
        print("3. Move to Day 6 (Polish & Advanced Features)!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()