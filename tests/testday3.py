# ============================================
# test_day3.py - Test Day 3 Features
# ============================================

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from layout_analyzer import LayoutAnalyzer, analyze_pdf_layout
from ocr_processor import OCRProcessor, extract_with_ocr_fallback

def test_layout_analysis():
    """Test layout detection"""
    print("=" * 70)
    print("TEST 1: LAYOUT ANALYSIS")
    print("=" * 70)
    
    # Get PDF
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 Testing with: {pdf_path.name}\n")
    
    # Analyze layout
    results = analyze_pdf_layout(str(pdf_path))
    
    # Display statistics
    print("\n📊 LAYOUT STATISTICS:")
    stats = results['stats']
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"\n  By type:")
    for block_type, count in stats['by_type'].items():
        print(f"    - {block_type}: {count}")
    
    # Show structure
    print(f"\n📋 DOCUMENT STRUCTURE:")
    structure = results['structure']
    print(f"  Headings found: {len(structure['headings'])}")
    print(f"  Paragraphs found: {len(structure['paragraphs'])}")
    
    # Show first 3 headings
    if structure['headings']:
        print(f"\n  Sample headings:")
        for heading in structure['headings'][:3]:
            print(f"    Page {heading['page']}: {heading['text'][:60]}")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    analyzer = results['analyzer']
    viz_path = analyzer.visualize_layout(page_num=1)
    if viz_path:
        print(f"  ✓ Saved to: {viz_path}")
    
    print("\n✅ Layout analysis test complete!")


def test_ocr():
    """Test OCR functionality"""
    print("\n" + "=" * 70)
    print("TEST 2: OCR CAPABILITY")
    print("=" * 70)
    
    # Get PDF
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 Testing with: {pdf_path.name}\n")
    
    # Initialize OCR
    ocr = OCRProcessor(use_gpu=True)
    
    # Check if PDF is scanned
    is_scanned = ocr.is_scanned_pdf(str(pdf_path))
    print(f"PDF type: {'Scanned (image-based)' if is_scanned else 'Native (text-based)'}")
    
    if is_scanned:
        print("\n🔍 Running OCR (this may take a few minutes)...")
        results = ocr.ocr_pdf(str(pdf_path), method='auto', languages=['eng'])
        
        # Show results
        print(f"\n📊 OCR RESULTS:")
        for page, data in list(results.items())[:3]:  # First 3 pages
            print(f"\n  Page {page}:")
            print(f"    Words: {data['word_count']}")
            print(f"    Confidence: {data['confidence']:.1f}%")
            print(f"    Preview: {data['text'][:100]}...")
    else:
        print("✓ No OCR needed for this PDF")
    
    print("\n✅ OCR test complete!")


def test_smart_extraction():
    """Test smart extraction with fallback"""
    print("\n" + "=" * 70)
    print("TEST 3: SMART EXTRACTION")
    print("=" * 70)
    
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 Testing with: {pdf_path.name}\n")
    
    # Smart extraction
    text = extract_with_ocr_fallback(str(pdf_path))
    
    # Statistics
    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"  Characters: {len(text):,}")
    print(f"  Words: {len(text.split()):,}")
    print(f"  Lines: {len(text.split(chr(10))):,}")
    
    # Preview
    print(f"\n  Preview (first 300 chars):")
    print(f"  {text[:300]}...")
    
    print("\n✅ Smart extraction test complete!")


def run_all_tests():
    """Run all Day 3 tests"""
    print("\n🧪 DAY 3 TESTING SUITE")
    print("=" * 70)
    
    try:
        test_layout_analysis()
        test_ocr()
        test_smart_extraction()
        
        print("\n" + "=" * 70)
        print("✅ ALL DAY 3 TESTS COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check layout_visualization.png")
        print("2. Review the output above")
        print("3. Ready for Day 4 (Summarization)!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()