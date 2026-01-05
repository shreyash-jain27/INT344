from src.layout_analyzer import LayoutAnalyzer
from pathlib import Path
import json

def test_layout_analysis():
    # Find a sample PDF
    pdf_files = list(Path("data/sample_reports").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found!")
        return

    pdf_path = str(pdf_files[0])
    print(f"Testing layout analysis on: {pdf_path}")

    analyzer = LayoutAnalyzer(pdf_path)
    layouts = analyzer.analyze_layout()

    print(f"✓ Analyzed {len(layouts)} pages")

    # Check first page
    if 1 in layouts:
        page1 = layouts[1]
        print(f"Page 1 Stats:")
        print(f"  - Dimensions: {page1['width']}x{page1['height']}")
        print(f"  - Text Blocks: {len(page1['blocks'])}")
        print(f"  - Images: {page1['image_count']}")
        
        # Check block types
        types = {}
        for block in page1['blocks']:
            t = block.get('type', 'Unknown')
            types[t] = types.get(t, 0) + 1
        print(f"  - Block Types: {types}")
        
        # Print first few blocks
        print("\nFirst 3 blocks:")
        for i, block in enumerate(page1['blocks'][:3]):
            print(f"  {i+1}. [{block['type']}] {block['text'][:50]}...")

if __name__ == "__main__":
    test_layout_analysis()
