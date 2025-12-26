# ============================================
# test_day4.py - Test summarization features
# ============================================

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from summarizer import UnifiedSummarizer
import time


def test_extractive():
    """Test extractive summarization"""
    print("=" * 70)
    print("TEST 1: EXTRACTIVE SUMMARIZATION")
    print("=" * 70)
    
    sample_text = """
    The company achieved record revenue of $10 billion in fiscal year 2023,
    representing a 25% increase from the previous year. This remarkable growth
    was driven by strong performance across all business segments, with
    technology and services leading the way. Our technology division saw
    particularly impressive results, with revenue growing 40% year-over-year,
    reaching $4 billion. The company continues to invest heavily in research
    and development, with R&D spending reaching $1.5 billion, an increase of
    20% from last year. We have launched several innovative products that have
    been extremely well-received by customers worldwide. Our market share has
    increased significantly in all major geographic regions, including North
    America, Europe, and Asia-Pacific. The board of directors has approved a
    15% increase in the dividend payout, reflecting our strong cash position
    and confidence in future prospects. Management remains highly optimistic
    about future growth prospects, citing strong customer demand, a robust
    pipeline of new products, and favorable market conditions. The company now
    employs over 50,000 people worldwide and operates in more than 100 countries.
    """
    
    from extractive_summarizer import ExtractiveSummarizer
    
    summarizer = ExtractiveSummarizer()
    
    print(f"\nOriginal: {len(sample_text.split())} words\n")
    
    methods = ['tfidf', 'textrank', 'bert']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"{method.upper()} Method:")
        print('='*50)
        
        start = time.time()
        summary = summarizer.summarize(sample_text, method=method, ratio=0.3)
        elapsed = time.time() - start
        
        print(f"\nSummary ({len(summary.split())} words):")
        print(summary)
        print(f"\nTime: {elapsed:.2f}s")
    
    print("\n✅ Extractive summarization test complete!")


def test_abstractive():
    """Test abstractive summarization"""
    print("\n" + "=" * 70)
    print("TEST 2: ABSTRACTIVE SUMMARIZATION")
    print("=" * 70)
    
    sample_text = """
    The company achieved record revenue of $10 billion in fiscal year 2023,
    representing a 25% increase from the previous year. This remarkable growth
    was driven by strong performance across all business segments, with
    technology and services leading the way. Our technology division saw
    particularly impressive results, with revenue growing 40% year-over-year,
    reaching $4 billion. The company continues to invest heavily in research
    and development, with R&D spending reaching $1.5 billion, an increase of
    20% from last year. We have launched several innovative products that have
    been extremely well-received by customers worldwide.
    """
    
    from abstractive_summarizer import AbstractiveSummarizer, get_best_model
    
    model = get_best_model()
    print(f"\nUsing model: {model}")
    print(f"Original: {len(sample_text.split())} words\n")
    
    summarizer = AbstractiveSummarizer(model_name=model)
    
    print("\nGenerating summary...")
    start = time.time()
    summary = summarizer.summarize(sample_text, max_length=80)
    elapsed = time.time() - start
    
    print(f"\nAbstractive Summary ({len(summary.split())} words):")
    print(summary)
    print(f"\nTime: {elapsed:.2f}s")
    print(f"Model info: {summarizer.get_model_info()}")
    
    print("\n✅ Abstractive summarization test complete!")


def test_unified():
    """Test unified summarizer"""
    print("\n" + "=" * 70)
    print("TEST 3: UNIFIED SUMMARIZER")
    print("=" * 70)
    
    sample_text = """
    The company achieved record revenue of $10 billion in fiscal year 2023,
    representing a 25% increase from the previous year. This remarkable growth
    was driven by strong performance across all business segments. Our technology
    division saw particularly impressive results, with revenue growing 40%
    year-over-year. The company continues to invest heavily in research and
    development, with R&D spending reaching $1.5 billion. We have launched
    several innovative products that have been extremely well-received by
    customers worldwide. Our market share has increased in all major geographic
    regions. The board has approved a 15% dividend increase. Management remains
    optimistic about future growth prospects.
    """
    
    summarizer = UnifiedSummarizer(use_gpu=True)
    
    print(f"\nOriginal: {len(sample_text.split())} words\n")
    
    methods = ['extractive', 'abstractive', 'hybrid']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"{method.upper()} Method:")
        print('='*50)
        
        result = summarizer.summarize(sample_text, method=method)
        
        print(f"\nSummary ({result.summary_length} words):")
        print(result.summary)
        print(f"\nStatistics:")
        print(f"  Method: {result.method}")
        print(f"  Compression: {result.compression_ratio*100:.1f}%")
        print(f"  Time: {result.processing_time:.2f}s")
        
        # Evaluate
        metrics = summarizer.evaluate_summary(sample_text, result.summary)
        print(f"  Readability: {metrics['flesch_reading_ease']:.1f}")
        print(f"  Grade Level: {metrics['flesch_kincaid_grade']:.1f}")
    
    print("\n✅ Unified summarizer test complete!")


def test_full_document():
    """Test with actual PDF"""
    print("\n" + "=" * 70)
    print("TEST 4: FULL DOCUMENT SUMMARIZATION")
    print("=" * 70)
    
    # Get PDF
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 Testing with: {pdf_path.name}\n")
    
    # Process PDF
    processor = PDFProcessor(str(pdf_path))
    sections = processor.identify_sections()
    
    if not sections:
        print("❌ No sections found")
        return
    
    print(f"✓ Found {len(sections)} sections")
    
    # Summarize
    summarizer = UnifiedSummarizer(use_gpu=True)
    
    # Pick one section to test
    test_section = list(sections.keys())[0]
    test_content = sections[test_section]
    
    print(f"\n📝 Summarizing: {test_section}")
    print(f"Original: {len(test_content.split())} words")
    
    result = summarizer.summarize(test_content, method='hybrid')
    
    print(f"\nSummary ({result.summary_length} words):")
    print(result.summary)
    print(f"\nCompression: {result.compression_ratio*100:.1f}%")
    print(f"Time: {result.processing_time:.2f}s")
    
    print("\n✅ Full document test complete!")


def test_section_batch():
    """Test batch summarization of sections"""
    print("\n" + "=" * 70)
    print("TEST 5: BATCH SECTION SUMMARIZATION")
    print("=" * 70)
    
    # Get PDF
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 Testing with: {pdf_path.name}\n")
    
    # Process PDF
    processor = PDFProcessor(str(pdf_path))
    sections = processor.identify_sections()
    
    if not sections:
        print("❌ No sections found")
        return
    
    # Take first 3 sections
    test_sections = dict(list(sections.items())[:3])
    
    print(f"Summarizing {len(test_sections)} sections...\n")
    
    # Summarize all
    summarizer = UnifiedSummarizer(use_gpu=True)
    results = summarizer.summarize_sections(test_sections, method='hybrid')
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for section_name, result in results.items():
        print(f"\n📄 {section_name}:")
        print(f"  Original: {result.original_length} words")
        print(f"  Summary: {result.summary_length} words")
        print(f"  Compression: {result.compression_ratio*100:.1f}%")
        print(f"  Time: {result.processing_time:.2f}s")
        print(f"\n  Summary preview:")
        print(f"  {result.summary[:200]}...")
    
    print("\n✅ Batch summarization test complete!")


def run_all_tests():
    """Run all Day 4 tests"""
    print("\n🧪 DAY 4 TESTING SUITE")
    print("=" * 70)
    print("\nThis will test all summarization methods.")
    print("Note: First run will download models (~2GB)")
    print("=" * 70)
    
    input("\nPress Enter to start...")
    
    try:
        test_extractive()
        test_abstractive()
        test_unified()
        test_full_document()
        test_section_batch()
        
        print("\n" + "=" * 70)
        print("✅ ALL DAY 4 TESTS COMPLETE!")
        print("=" * 70)
        print("\n🎉 Your summarizer is working!")
        print("\nNext steps:")
        print("1. Check the summaries above")
        print("2. Try with different PDFs")
        print("3. Integrate into web interface")
        print("4. Move to Day 5 (Multilingual Support)!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()