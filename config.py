import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    BASE_URL = "https://newsapi.org/v2/top-headlines"
    DEFAULT_CATEGORIES = ['general']
    COUNTRIES = {'us': 'United States', 'gb': 'UK', 'in': 'India'}
    CACHE_TIMEOUT = 1800  # 30 minutes