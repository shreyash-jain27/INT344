# 🔍 Granular File Explanations (Line-by-Line Breakdown)

This guide provides a deep dive into the logic of every major file in the project. Use this to answer specific "what does this line do?" questions.

---

## 1. `app.py` (Streamlit Frontend)

**Key Logic Block: File Upload & Persistence**
```python
if uploaded_file:
    temp_path = Path("temp_upload.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
```
- **Meaning:** Converts the browser's "uploaded file object" into a real physical file on the computer (`temp_upload.pdf`) so our other Python scripts can read it using standard file paths.

**Key Logic Block: The Tabbed Interface**
```python
tabs = st.tabs(["📄 Overview", "📑 Sections", ...])
with tabs[0]: 
    st.header("Document Overview")
```
- **Meaning:** Organizes the complex data into user-friendly pages. Each `with tabs[i]` block contains the code logic for that specific feature.

---

## 2. `src/pdf_processor.py` (The Heavy Lifter)

**Key Logic Block: Text Extraction with `pdfplumber`**
```python
with pdfplumber.open(self.pdf_path) as pdf:
   for page in pdf.pages:
       text += page.extract_text() or ""
```
- **Meaning:** Opens the PDF and iterates through every page. It grabs every bit of text it can find. The `or ""` handles blank pages (which would otherwise cause a crash).

**Key Logic Block: Section Identification**
```python
for name, keywords in self.SECTION_KEYWORDS.items():
   if any(k.lower() in page_text.lower() for k in keywords):
       sections[name] += page_text
```
- **Meaning:** We look for specific "trigger words" (like "Balance Sheet"). If a page has that word, we assign that entire page's content to that section heading.

---

## 3. `src/abstractive_summarizer.py` (The Summarization AI)

**Key Logic Block: Tokenization**
```python
inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
```
- **Meaning:**
    - `return_tensors="pt"`: Formats the data for **PyTorch**.
    - `max_length=1024`: Cuts off the text if it's too long (models have limits).
    - `truncation=True`: Ensures the program doesn't crash if the text is too long.

**Key Logic Block: Abstractive Generation**
```python
summary_ids = self.model.generate(input_ids, num_beams=4, length_penalty=2.0)
```
- **Meaning:**
    - `num_beams=4`: Explores multiple word possibilities simultaneously.
    - `length_penalty=2.0`: Forces the model to create **longer**, more detailed summaries (higher penalty = more words).

---

## 4. `src/translator.py` (The Translation Hub)

**Key Logic Block: Local Model Loading**
```python
self.punjabi_model = AutoModelForSeq2SeqLM.from_pretrained(self.local_model_path)
```
- **Meaning:** Instead of downloading a model from the internet, it loads the model from your `models/punjabi_translator` folder. This is why it works offline!

**Key Logic Block: Translation with Google**
```python
translator = GoogleTranslator(source=source_code, target=target_code)
translated = translator.translate(chunk)
```
- **Meaning:** Creates a "translator object" for the specific language pair and sends a small chunk of text to the cloud to be translated.

---

## 5. `src/language_detector.py` (The Logic)

**Key Logic Block: Unicode Script Check**
```python
for char in text:
    code_point = ord(char)
    if 0x0A00 <= code_point <= 0x0A7F: # Punjabi
        script_counts['pa'] += 1
```
- **Meaning:** Every character in the world has a code number (Unicode). Punjabi characters always live between `0x0A00` and `0x0A7F`. We simply count how many characters fall in that range to identify the language.

---

## 📂 Summary of "How files work with each other"
1. **User uploads** in `app.py`.
2. `app.py` passes the file path to `PDFProcessor` (`src/pdf_processor.py`).
3. `PDFProcessor` uses `LayoutAnalyzer` (`src/layout_analyzer.py`) to find structure.
4. When the user selects a language, `MultilingualSummarizer` (`src/multilingual_summarizer.py`) kicks in.
5. It uses `LanguageDetector` (`src/language_detector.py`) to confirm the input language.
6. It then asks `Translator` (`src/translator.py`) and `AbstractiveSummarizer` (`src/abstractive_summarizer.py`) to do the work.
勾
