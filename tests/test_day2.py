"""
Create this file in your main project folder (not in src)
"""

from src.pdf_processor import PDFProcessor
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_pdf_processing():
    """Test the PDF processor with your annual report"""

    print("=" * 60)
    print("DAY 2: TESTING PDF PROCESSOR")
    print("=" * 60)

    # List available PDFs
    pdf_dir = Path("data/sample_reports")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data/sample_reports/")
        print("Please add some annual report PDFs and try again!")
        return

    print(f"\n📁 Found {len(pdf_files)} PDF file(s):")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file.name}")

    # Process the first PDF (you can change this)
    test_pdf = pdf_files[0]
    print(f"\n🔬 Testing with: {test_pdf.name}")
    print("-" * 60)

    # Create processor
    processor = PDFProcessor(str(test_pdf))

    # Extract text
    text = processor.extract_text()

    # Identify sections
    sections = processor.identify_sections()

    # Extract tables
    tables = processor.extract_tables()

    # Get statistics
    stats = processor.get_section_stats()

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n📊 SECTIONS FOUND ({len(sections)}):")
    for i, (section_name, content) in enumerate(sections.items(), 1):
        word_count = stats[section_name]['word_count']
        print(f"  {i}. {section_name}")
        print(f"     - Words: {word_count:,}")
        print(f"     - Preview: {content[:100]}...")
        print()

    print(f"📋 TABLES FOUND: {len(tables)}")
    if tables:
        for table in tables[:3]:  # Show first 3 tables
            print(
                f"  - Page {table['page']}: {table['rows']}x{table['columns']} table")

    # Save sections to files
    output_dir = processor.save_sections()

    print("\n" + "=" * 60)
    print("✅ DAY 2 COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Check your results in: {output_dir}")
    print("\nNext: Run 'streamlit run app.py' to see the web interface!")


if __name__ == "__main__":
    test_pdf_processing()
