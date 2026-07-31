import nltk

print("Downloading required NLTK datasets...")
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK datasets downloaded successfully!")
except Exception as e:
    print(f"Error downloading programmatically: {e}")
    print("Opening interactive download manager...")
    nltk.download()
