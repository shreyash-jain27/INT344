from src.multilingual import MultilingualHandler
import time

def test_multilingual():
    print("Initializing MultilingualHandler...")
    handler = MultilingualHandler()
    
    # Test Language Detection
    print("\n--- Testing Language Detection ---")
    texts = {
        "en": "This is a sample text in English.",
        "hi": "यह हिंदी में एक नमूना पाठ है।",
        "ta": "இது தமிழில் ஒரு மாதிரி உரை."
    }
    
    for lang, text in texts.items():
        detected = handler.detect_language(text)
        print(f"Text: {text[:30]}... | Expected: {lang} | Detected: {detected}")
        
    # Test Translation
    print("\n--- Testing Translation ---")
    text = "The company has achieved significant growth this year."
    targets = ['hi', 'ta', 'bn']
    
    print(f"Original: {text}")
    
    for target in targets:
        print(f"Translating to {target}...")
        translated = handler.translate(text, dest_lang=target)
        print(f"  [{target}]: {translated}")
        
if __name__ == "__main__":
    test_multilingual()
