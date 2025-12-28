# 💻 Technical Code Deconstruction: How the Engine Works

This file explains the logic of the core code blocks so you can explain exactly "how" the code is doing what it says.

---

## 1. `app.py` (The Control Center)

### The "Session State" Logic
We use `st.session_state` everywhere. 
**Why?** Streamlit reruns the whole script every time you click a button. Without session state, all your variables (like the uploaded PDF or the summary) would be deleted.
```python
if 'doc_data' not in st.session_state:
    st.session_state['doc_data'] = process_document(processor)
```
*Meaning: "If we haven't analyzed this document yet, do it now and store it in a permanent memory bucket so it doesn't disappear on the next click."*

---

## 2. `src/pdf_processor.py` (Text Extraction)

### Section Identification
How does the computer know what is the "Chairman's Speech"?
```python
def identify_sections(self):
    for name, keywords in self.SECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in page_text.lower():
                sections[name] += page_text
```
*Logic: We define a dictionary of keywords (e.g., 'Chairman', 'Message', 'Speech'). We loop through every page. If the page contains any of those words, we "tag" that page to that specific section.*

---

## 3. `src/abstractive_summarizer.py` (AI Brain)

### The Chunking Mechanism
This is the most complex part. Models have a `max_position_embeddings` (usually 1024 tokens).
```python
def _chunk_text(self, text, max_tokens=800):
   # ... splits text by paragraphs ...
   if current_chunk_size + delta > max_tokens:
       chunks.append(" ".join(current_chunk))
```
*Logic: We monitor the number of words. When a chunk reaches ~800 words, we "close" it and start a new one. This ensures the AI model never gets "cut off" mid-sentence.*

### Model Generation
```python
summary_ids = self.model.generate(
    inputs['input_ids'],
    max_length=150,
    num_beams=4,
    early_stopping=True
)
```
*Meaning:*
- `input_ids`: The numbers the text was turned into.
- `num_beams=4`: "Beam Search". The model doesn't just pick the single best next word; it tracks the top 4 most likely paths/sentences and chooses the one that makes the most sense globally.

---

## 4. `src/translator.py` (The Local Punjabi AI)

### The language tag `>>pan<<`
Our Punjabi model is based on "Helsinki-NLP". It is a **Many-to-Many** model. 
```python
input_text = f">>pan<< {text}"
```
*Logic: The model was trained on dozens of languages. We must prepend `>>pan<<` (Punjabi tag) to the beginning of our English string to "activate" the Punjabi translation neurons in the model.*

---

## 5. `src/layout_analyzer.py` (Geometric Math)

### The Scaling Formula
If the PDF is 600 points wide and the Image is 1200 pixels wide, our scaling factor is 2.0.
```python
scale_x = img.width / float(p.width)
bbox = [block.bbox[0] * scale_x, ...]
```
*Logic: We take the PDF "Coordinate" (e.g., x=100) and multiply it by our scale to find the "Pixel" coordinate on the screen. This is why the red boxes line up perfectly with the text.*

---

## 📂 Summary of Interaction
1. **User interacts** with `app.py`.
2. `app.py` creates an instance of `PDFProcessor`.
3. `PDFProcessor` creates an instance of `LayoutAnalyzer`.
4. When "Summarize" is clicked, `app.py` calls `MultilingualSummarizer`.
5. `MultilingualSummarizer` asks `LanguageDetector` "What is this?" and then asks `Translator` "Translate this to English".
6. It then asks `AbstractiveSummarizer` "Summarize this English part".
7. Finally, it asks `Translator` "Translate it back to Punjabi".

---

## 💡 Viva Tip: If you forget a term...
- **Tokenization:** Turning sentences into chunks/numbers for AI.
- **Inference:** The process of a model generating an answer.
- **Fine-tuning:** Taking a "smart" general model and teaching it a specific skill (like Punjabi).
- **Embeddings:** Turning words into geometry (vectors) so the computer can calculate "meaning".
勾
