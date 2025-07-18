# === INSTALLATION (run this in terminal or notebook first) ===
# pip install requests python-dotenv newspaper3k beautifulsoup4 nltk unidecode

import os
import json
import re
import string
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

import requests
from newspaper import Article
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from dotenv import load_dotenv
import nltk

# === SETUP ===
try:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")
except Exception as e:
    print(f"❌ NLTK downloads failed: {e}")
    raise

load_dotenv()
API_KEY = os.getenv("MEDIASTACK_API_KEY")
ENDPOINT = "http://api.mediastack.com/v1/news"

# Output file
OUTPUT_FOLDER = "data/raw/mediastack"
MASTER_PATH = f"{OUTPUT_FOLDER}/all_mediastack_articles_2025_5.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# NLP setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
REFERENCE_DATE = datetime(2025, 5, 12, tzinfo=timezone.utc)

# === SMART CLEANING ===
def is_noise(sentence: str) -> bool:
    """Check if a sentence contains common noise patterns."""
    noise_keywords = [
        "subscribe", "sign up", "follow us", "download the app", "photo by",
        "reporting by", "read more", "share this", "get the app", "contact us",
        "click here", "read our", "help us improve"
    ]
    sentence = sentence.lower()
    return any(kw in sentence for kw in noise_keywords)

def clean_text(text: str) -> str:
    """Clean and preprocess text for NLP analysis."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = unidecode(text)

    # Sentence-level filtering
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s for s in sentences if not is_noise(s)]
    text = " ".join(sentences)

    # Remove URLs, numbers, punctuation
    text = re.sub(r"http\S+|www\S+|\S+@\S+", "", text)
    text = re.sub(r"[0-9]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)

def fetch_articles_for_date(date_str: str) -> List[Dict]:
    """Fetch articles from MediaStack API for a specific date."""
    params = {
        "access_key": API_KEY,
        "languages": "en",
        "date": date_str,
        "limit": 100,
    }
    
    try:
        response = requests.get(ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data.get("data"), list):
            print(f"⚠️ Unexpected API response format for {date_str}")
            return []
            
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed for {date_str}: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Failed to decode API response for {date_str}: {e}")
        return []

def process_article(item: Dict) -> Optional[Dict]:
    """Process individual article from API response."""
    try:
        url = item["url"]
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        pub_dt = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
        t_value = (REFERENCE_DATE - pub_dt).days

        return {
            "source": item.get("source", ""),
            "url": url,
            "date": pub_dt.strftime("%Y-%m-%d"),
            "time": pub_dt.strftime("%H:%M:%S"),
            "title": article.title,
            "body": article.text,
            "clean_body": clean_text(article.text),
            "summary": item.get("description", article.summary),
            "keywords": article.keywords,
            "image_url": item.get("image", article.top_image),
            "category": item.get("category", ""),
            "t": t_value
        }
    except Exception as e:
        print(f"❌ Failed to process {item.get('url', 'unknown')} - {str(e)}")
        return None

def scrape_range(start_date: datetime, end_date: datetime) -> None:
    """Main function to scrape articles between date ranges."""
    if not API_KEY:
        raise ValueError("MEDIASTACK_API_KEY not found in environment variables")

    total_processed = 0
    total_saved = 0
    master_articles = {}

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        print(f"📆 Fetching articles for {date_str}...")
        
        items = fetch_articles_for_date(date_str)
        processed_articles = [process_article(item) for item in items]
        valid_articles = [a for a in processed_articles if a is not None]
        
        total_processed += len(items)
        total_saved += len(valid_articles)

        if valid_articles:
            # Use URL as key to avoid duplicates
            for article in valid_articles:
                master_articles[article["url"]] = article
            print(f"✅ Collected {len(valid_articles)} articles")
        else:
            print(f"⚠️ No valid articles for {date_str}")

        current += timedelta(days=1)
        time.sleep(1)  # Rate limiting

    # Save results
    with open(MASTER_PATH, "w", encoding="utf-8") as f:
        json.dump(list(master_articles.values()), f, indent=2, ensure_ascii=False)

    print(f"\n📊 Total processed: {total_processed}")
    print(f"📦 Total saved: {total_saved}")
    print(f"💾 File saved to: {MASTER_PATH}")

# === ENTRY POINT ===
if __name__ == "__main__":
    try:
        # Adjust date range as needed
        scrape_range(
            start_date=datetime(2025, 5, 1),
            end_date=datetime(2025, 5, 12)
        )
    except Exception as e:
        print(f"❌ Script failed: {e}")