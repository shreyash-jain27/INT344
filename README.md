# Layout-Aware Summarization of Annual Reports for Multilingual India (v1.0 Stable)

An intelligent, robust system to extract, analyze, and summarize Indian annual reports with multilingual support and high-performance document processing.

## 🚀 Version 1.0 Highlights

- **Optimized Performance**: Single-pass document analysis with localized caching and instant tab switching.
- **Lazy Loading**: App startup is nearly instant; heavy AI models are loaded only when requested.
- **Improved Stability**: Robust error boundaries prevent crashes during model loading or processing failures.
- **Memory Management**: Automatic garbage collection and session purging when switching documents.

## 🛠️ Features

- **Advanced PDF Processing**: High-fidelity text extraction using multiple engines (`pdfplumber`, `PyPDF2`).
- **Layout Analysis**: Page structure visualization with bounding box identification.
- **Multilingual Support**:
  - Automatic Indian language detection.
  - Summarization for Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, and Assamese.
  - **MBart-50 Support**: High-performance local translation and summarization using `facebook/mbart-large-50-many-to-many-mmt`.
  - **Hybrid Translation Strategy**:
    - Uses **MBart-50** for Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, and Malayalam.
    - Falls back to **Google Translate** for languages not natively supported by MBart-50 (Kannada, Punjabi, Odia, Assamese).
- **Interactive Visuals**: Plotly charts for section distribution and keyword frequency.
- **Export Options**: Professional DOCX, CSV, and Text reports.

## 🛠️ Installation & Setup

1. **Clone & Environment**
   ```bash
   git clone <repository-url>
   cd annual_report_summarizer
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate
   ```

2. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Resources**
   - Ensure `tesseract` is installed for OCR features.
   - The app will automatically download ~2.3GB of AI models (MBart-50) on the first run of "Direct" multilingual summarization.

## 🏃 Usage

```bash
streamlit run app.py
```

## 🔍 Troubleshooting

- **indic_nlp_resources**: If you see errors about missing Indic resources, ensure the `indic_nlp_resources` folder is present in the root directory.
- **Memory Issues**: For large PDFs (>100 pages), use a machine with at least 8GB of RAM if using direct multilingual models.
- **Model Vulnerability**: If you see `torch.load` warnings, this stable version has been patched to use `safetensors` for all AI model loading.

## 📁 Project Structure

```
annual_report_summarizer/
├── src/
│   ├── pdf_processor.py         # Advanced PDF extraction & logic
│   ├── layout_analyzer.py       # Visual structure analysis
│   ├── multilingual_summarizer.py # BART-based Indian language support
│   ├── language_detector.py    # Auto-detection for 10+ languages
│   ├── translator.py           # MBart-50 & Google fallbacks
│   └── summarizer.py           # Extractive & Abstractive engines
├── app.py                      # Main Streamlit interface
└── requirements.txt            # Project dependencies
```

## 📄 License
MIT License - 2025
