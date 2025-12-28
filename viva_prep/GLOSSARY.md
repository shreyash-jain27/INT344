# 📖 Technical Glossary (Viva Cheat Sheet)

If the examiner throws around "big words," use this glossary to explain what they mean in the context of your project.

---

### 🧠 Artificial Intelligence & NLP
1. **NLP (Natural Language Processing):** The branch of AI that helps computers understand, interpret, and generate human language.
2. **Transformer:** The specific "type" of AI architecture we used (e.g., BART). It uses a mechanism called "Attention" to weigh the importance of different words in a sentence.
3. **LLM (Large Language Model):** AI trained on massive amounts of text. Models like BART are the "engine" inside our app.
4. **Tokenization:** The process of breaking a sentence into smaller pieces (tokens) like words or sub-words so the AI can process them.
5. **Inference:** The actual act of the AI "thinking" and generating an answer (e.g., when you click summarize).

---

### 📝 Summarization Concepts
6. **Abstractive Summarization:** The AI reads the text and **rewrites** it in its own words (like a human would). *Hardware intensive, but high quality.*
7. **Extractive Summarization:** The computer identifies the most important **existing sentences** and groups them together. *Fast and accurate.*
8. **Hybrid Summarization:** Using **both** methods together to get the best of both worlds.
9. **ROUGE Score:** A metric used to evaluate summaries by comparing how many words overlap between the AI summary and a human-written one.

---

### 🌍 Translation & Multilingual
10. **Fine-tuning:** Taking a pre-trained model (like Opus-MT) and training it further on a specific dataset (like Punjabi) to make it an expert in that area.
11. **IndicBART:** A special version of BART trained specifically on 11 Indian languages.
12. **BLEU Score:** The industry-standard metric for translation. It measures how "human-like" the machine translation is.
13. **Language Detection:** Using the "Unicode Script" (the unique digital fingerprint of Gurmukhi or Devanagari) to identify the language.

---

### 💻 System & Engineering
14. **Lazy Loading:** A coding trick where we only load heavy libraries (like Torch) when the user clicks a button, rather than at the start. *Ensures app stability.*
15. **Session State:** A way to keep variables alive in Streamlit even when the web page refreshes.
16. **OCR (Optical Character Recognition):** Technology that converts images of text (scanned PDFs) into real editable text.
17. **Vector Embeddings:** Turning sentences into lists of numbers (coordinates) so the computer can calculate how "close" two sentences are in meaning.
勾
