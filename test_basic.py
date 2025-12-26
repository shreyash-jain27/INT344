import PyPDF2
import pdfplumber

# Test 1: Check if PDF can be read
# Change this to your PDF name
pdf_path = "data/sample_reports/AR_26481_INFY_2024_2025_A_02062025153945.pdf"

print("Testing PDF reading...")
with pdfplumber.open(pdf_path) as pdf:
    print(f"✓ PDF has {len(pdf.pages)} pages")

    # Extract first page
    first_page = pdf.pages[0]
    text = first_page.extract_text()

    print(f"✓ First page text length: {len(text)} characters")
    print("\n--- First 500 characters ---")
    print(text[:500])

print("\n✓ Basic PDF processing works!")
