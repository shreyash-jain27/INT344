from src.summarizer import Summarizer
import time

def test_summarization():
    print("Initializing Summarizer...")
    summarizer = Summarizer()
    
    # Sample text (Chairman's message style)
    text = """
    Dear Shareholders,
    
    It gives me immense pleasure to present the Annual Report of your Company for the financial year 2023-24. 
    Despite global headwinds and geopolitical challenges, your Company has delivered a resilient performance. 
    We achieved a revenue growth of 15% year-on-year, reaching an all-time high of ₹50,000 Crores.
    Our profit after tax stood at ₹5,000 Crores, reflecting a healthy margin of 10%.
    
    We have continued to invest in technology and innovation. Our digital transformation journey is yielding significant results.
    We launched 5 new products this year, which have been well received in the market.
    Sustainability remains at the core of our strategy. We have reduced our carbon footprint by 20% and are on track to become carbon neutral by 2030.
    
    I would like to thank all our employees for their dedication and hard work. 
    I also express my gratitude to our shareholders, customers, and partners for their continued trust and support.
    
    The future looks promising, and we are well-positioned to capture new opportunities.
    """
    
    print(f"\nOriginal Text Length: {len(text)} chars")
    
    # Test Extractive
    print("\n--- Testing Extractive Summarization ---")
    start = time.time()
    ext_summary = summarizer.extractive_summary(text, sentences_count=2)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Summary: {ext_summary}")
    
    # Test Abstractive
    print("\n--- Testing Abstractive Summarization ---")
    start = time.time()
    abs_summary = summarizer.abstractive_summary(text, max_length=100)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Summary: {abs_summary}")
    
    # Test Hybrid
    print("\n--- Testing Hybrid Summarization ---")
    start = time.time()
    hyb_summary = summarizer.hybrid_summary(text, final_max_length=100)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Summary: {hyb_summary}")

if __name__ == "__main__":
    test_summarization()
