# 🎓 Comprehensive Viva Guide: Annual Report Summarizer

This guide provides a complete breakdown of the project, including the core concepts, technical implementation, and how the different components work together. Study this to gain complete mastery over the project you've built.

---

## 1. Project Overview
**What is this?** An AI-powered platform to analyze, summarize, and translate corporate Annual Reports (PDFs). 
**Problem it solves:** Annual reports are hundreds of pages long and often in English. Shareholders need quick insights and often prefer their native Indian language.
**Key Features:**
- **Layout Intelligence:** Identifies headings, paragraphs, and tables.
- **Hybrid Summarization:** Combines statistical (speed) and AI (quality) methods.
- **Multilingual Support:** Summarizes and translates across 11+ Indian languages.
- **Custom Punjabi Translator:** A fine-tuned model specifically for English-to-Punjabi translation.

---

## 2. The Tech Stack
- **Dashboard:** Streamlit (Python-based web library).
- **Core NLP:** Hugging Face Transformers (BART, IndicBART, Opus-MT).
- **Computer Vision:** `pdfplumber` (text extraction) and `pdf2image`/`PIL` for layout visualization.
- **Translation:** `deep-translator` (Google API) and local transformer models.
- **Preprocessing:** `IndicNLP` library for Indian language normalization and tokenization.

---

## 3. Directory & File Breakdown

### 📂 Root Directory
- `app.py`: **The Heart**. The Streamlit frontend. It coordinates the user's PDF upload, clicks, and displays.
- `train_punjabi_translator.ipynb`: The **Training Lab**. Used to fine-tune the `Helsinki-NLP/opus-mt-en-inc` model on the `opus100` Punjabi dataset.
- `venv/`: The **Storage**. Contains all Python libraries installed (Torch, Transformers, etc.).

### 📂 `src/` (The Engine)
This is where the actual logic lives. Every file has a specific responsibility.

#### 1. `pdf_processor.py` (The Gatekeeper)
- **Purpose:** Acts as the high-level orchestrator. It uses `pdfplumber` to extract raw text and then calls other "specialists" to analyze further.
- **Logic:** It "chunks" the document into sections (Chairman's Speech, Financial statements) based on keyword matching.

#### 2. `summarizer.py` (The English Specialist)
- **Purpose:** Controls English summarization.
- **Key Method:** `summarize()`. It decides whether to use **Extractive** (picking best sentences) or **Abstractive** (writing new summary) models based on user choice.

#### 3. `abstractive_summarizer.py` (The AI Writer)
- **Purpose:** Uses Deep Learning (BART/T5) to **rewrite** text.
- **Mechanism:** It tokenizes text, passes it through a Transformer, and generates new sentences.
- **Vital Implementation:** `Chunked Summarization`. Since models can't read 100 pages at once, this file splits text into chunks, summarizes each, and merges them.

#### 4. `extractive_summarizer.py` (The Text Picker)
- **Purpose:** Uses statistics to find the most important existing sentences.
- **Mechanism:** Uses `Sentence-Transformers` (BERT) to turn sentences into numbers (embeddings/vectors) and picks the ones most central to the topic.

#### 5. `multilingual_summarizer.py` (The Cross-Lingual Hub)
- **Purpose:** Handles the Indian language "Translate-Summarize-Translate" workflow.
- **Logic:** Calls `translator.py` to convert Indian text to English -> summarizes in English -> translates back. Also supports direct summarization via **IndicBART**.

#### 6. `translator.py` (The Linguist)
- **Purpose:** Handles all translation logic.
- **Three Pillars:**
  1. **Google Translate:** High reliability for general text.
  2. **IndicBART:** Transformer model for Indian-to-Indian translation.
  3. **Local Punjabi Model:** A custom model we trained specifically for the user's Punjabi requirement.

#### 7. `language_detector.py` (The Identifier)
- **Purpose:** Tells the system what language is being read.
- **Concept:** Uses **Unicode Script Detection**. (e.g., If characters fall in range `0x0A00` to `0x0A7F`, it's Punjabi). This is much more accurate for Indian languages than simple probability.

#### 8. `layout_analyzer.py` (The Eye)
- **Purpose:** Understands the "Shape" of the document.
- **Mechanism:** Detects font sizes and positions. If text is large and at the top, it's a **Heading**. If it's small blocks, it's a **Paragraph**.

---

## 4. Crucial Concept: "Lazy Loading" 🛡️
You will likely be asked why your imports are inside functions.
- **The Problem:** In a Viva, you might notice Streamlit crashing with a "RuntimeError" if you change code while it's running. This happens because Streamlit's "Watcher" gets overwhelmed by huge libraries like `torch`.
- **The Solution:** We implemented **Lazy Loading**. Python only "imports" `torch` or `transformers` at the exact moment the user clicks "Summarize". This keeps the app incredibly stable and fast to launch.

---

## 5. The Punjabi Model training process
Explain this step-by-step for the ML portion of the viva:
1. **Dataset:** Used `opus100` (English-Punjabi pairs).
2. **Tokenizer:** Preprocessed text using **SentencePiece** to handle the Gurmukhi script.
3. **Training Log:** Used a `Seq2SeqTrainer` from Hugging Face. Prepended `>>pan<<` to every English sentence to tell the model the target is Punjabi.
4. **Evaluation:** Measured success using **BLEU Score** (measures how close the text is to a human reference).

---

## 6. How it all works together (The Workflow)
1. **Upload:** User provides a PDF.
2. **Analysis:** `pdf_processor` extracts text; `layout_analyzer` maps out the headings.
3. **Detection:** `language_detector` checks the language of each section.
4. **Summarization:**
   - If **English**: Call `summarizer.py`.
   - If **Punjabi/Hindi**: Call `multilingual_summarizer.py`.
5. **Translation:** The multilingual hub uses `translator.py` to handle the heavy lifting.
6. **Visualization:** The `app.py` displays Plotly charts for statistics and images for layout.

---

## 7. Potential Viva Questions & Answers

**Q: Why use BART for summarization?**
*A: BART is a denoising autoencoder. It's excellent at "reconstructing" a concise version of a corrupted/long input text, making it perfect for abstractive summarization.*

**Q: How do you handle PDFs that are just images?**
*A: We have an `ocr_processor.py` that uses Tesseract and EasyOCR to "see" text inside images and convert it back to searchable strings.*

**Q: What is the "Translate-Summarize-Translate" approach?**
*A: English summarization models are the most advanced. By translating Hindi to English, summarizing, and translating back, we get higher quality summaries than trying to summarize Hindi directly.*

**Q: What is "IndicBART"?**
*A: It's a multilingual model pre-trained on 11 Indian languages. Unlike standard BART, it understands the unique syntax and grammar of Indian scripts.*
