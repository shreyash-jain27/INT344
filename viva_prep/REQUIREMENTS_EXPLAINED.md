# 📦 Library & Dependency Guide: What does what?

If the examiner asks: "Why do you have 50 libraries installed?", use this list to explain the heavy hitters.

### 1. `torch` (PyTorch)
- **The Engine.** This is the massive library that actually runs the neural networks. All our AI models (BART, IndicBART, Punjabi model) run on top of PyTorch.

### 2. `transformers` (by Hugging Face)
- **The Library.** This gives us the high-level code to load and use models like "BART" without writing the complex math from scratch.

### 3. `pdfplumber`
- **The Extractor.** Better than standard PyPDF2 because it extracts "words" with their coordinates (x,y), which allows us to do Layout Analysis.

### 4. `deep-translator`
- **The Connector.** This connects our app to the Google Translate API cloud. It handles the splitting of text into 5000-character chunks to avoid API limits.

### 5. `IndicNLP`
- **The Specialist.** Essential for Indian languages. It handles script "normalization" (ensuring different ways of typing the same character are treated as one).

### 6. `plotly`
- **The Artist.** Used to create those interactive bar charts and pie charts in the "Overview" and "Visuals" tabs.

### 7. `nltk` (Natural Language Toolkit)
- **The Classic.** Used for basic stuff like splitting text into sentences (tokenization) and measuring word counts.

### 8. `wordcloud`
- **The Visualizer.** Specifically used to create the "Word Cloud" images showing the most frequent terms in a section.

---

### Command to set up the environment (if asked):
`pip install streamlit torch transformers pdfplumber deep-translator indic-nlp-library nltk plotly wordcloud pdf2image openpyxl`
