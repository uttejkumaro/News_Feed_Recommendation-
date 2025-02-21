import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'  # Disable file watcher
import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
import plotly.express as px
from dotenv import load_dotenv
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article
import sqlite3
from googletrans import Translator, LANGUAGES
import logging
from cachetools import TTLCache
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/top-headlines"

# Initialize cache (TTL: 300 seconds)
cache = TTLCache(maxsize=100, ttl=300)

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize translator
translator = Translator()

# Database setup
def init_db():
    conn = sqlite3.connect('newsflow.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_preferences 
                 (user_id INTEGER, category TEXT, country TEXT, sentiment TEXT, FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# User authentication
def login_page():
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Submit", key="login_submit"):
        conn = sqlite3.connect('newsflow.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            st.session_state.user = username
            st.sidebar.success("Logged in successfully!")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials!")

def signup_page():
    st.sidebar.subheader("Sign Up")
    new_username = st.sidebar.text_input("New Username", key="signup_username")
    new_password = st.sidebar.text_input("New Password", type="password", key="signup_password")
    if st.sidebar.button("Create Account", key="signup_submit"):
        conn = sqlite3.connect('newsflow.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, new_password))
            conn.commit()
            st.sidebar.success("Account created successfully! Please log in.")
            st.session_state.signup_complete = True
        except sqlite3.IntegrityError:
            st.sidebar.error("Username already exists!")
        finally:
            conn.close()

def auth_flow():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'signup_complete' not in st.session_state:
        st.session_state.signup_complete = False

    if st.session_state.user is None:
        auth_choice = st.sidebar.radio("Choose an option", ["Login", "Sign Up"], key="auth_choice")
        if auth_choice == "Login":
            login_page()
        elif auth_choice == "Sign Up":
            signup_page()
            if st.session_state.signup_complete:
                login_page()
        return False
    return True

# Fetch news with caching
@st.cache_data
def fetch_news(params):
    params['apiKey'] = NEWS_API_KEY
    cache_key = str(params)
    if cache_key in cache:
        return cache[cache_key]
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            cache[cache_key] = articles
            return articles
        else:
            logger.error(f"News API Error: {data.get('message', 'Unknown Error')}")
            st.error(f"News API Error: {data.get('message', 'Unknown Error')}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        st.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(text):
    if not text:
        return "neutral"
    try:
        text = str(text) if text else ""
        doc = nlp(text)
        sentiment = TextBlob(text).sentiment.polarity
        return 'positive' if sentiment > 0.1 else 'negative' if sentiment < -0.1 else 'neutral'
    except Exception as e:
        logger.error(f"Sentiment Analysis Error: {e}")
        return "neutral"

def get_trending_topics(articles):
    all_words = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        combined_text = f"{title} {description}".strip()
        if combined_text:
            doc = nlp(combined_text)
            text = " ".join([token.text.lower() for token in doc 
                           if not token.is_stop and token.is_alpha])
            all_words.extend(text.split())
    word_counts = Counter(all_words)
    return word_counts.most_common(10)

def extract_keywords(text):
    if not text:
        return []
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return Counter(keywords).most_common(5)

def summarize_article(url):
    if url in cache:
        return cache[url]
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        summary = article.summary
        cache[url] = summary
        return summary
    except Exception as e:
        logger.error(f"Article Summarization Error: {e}")
        return None

def recommend_articles(articles, user_history):
    user_keywords = set()
    if user_history:
        for entry in user_history:
            url = entry.get('url')
            if url and url in st.session_state.user_profile.get('article_keywords', {}):
                user_keywords.update([word for word, _ in st.session_state.user_profile['article_keywords'][url]])
    
    scores = []
    for article in articles or []:
        article_keywords = set([word for word, _ in extract_keywords(article.get('description', ''))])
        overlap = len(user_keywords.intersection(article_keywords)) if user_keywords else 0
        scores.append((article, overlap))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:10]

def display_article(article, idx, user):
    article_url = article.get('url')
    if not article or not article_url:
        return

    with st.expander(f"{idx + 1}. {article.get('title', 'No Title')}"):
        st.markdown(f"**Source**: {article.get('source', {}).get('name', 'Unknown')}")
        published_at = article.get('publishedAt')
        st.markdown(f"**Published**: {published_at[:10] if published_at else 'Unknown'}")
        st.markdown(f"**Sentiment**: {article.get('sentiment', 'Unknown').capitalize()}")

        image_url = article.get('urlToImage')
        if image_url:
            st.image(image_url, width=300, caption=article.get('description', 'No Description'), use_container_width=True)

        st.markdown(f"[Read full article]({article_url})")

        keywords = extract_keywords(article.get('description', ''))
        st.write("**Keywords:**", ", ".join([word for word, _ in keywords]))

        if article_url not in st.session_state.user_profile.get('article_summaries', {}):
            with st.spinner("Summarizing article..."):
                summary = summarize_article(article_url)
                if summary:
                    st.session_state.user_profile.setdefault('article_summaries', {})[article_url] = summary

        summary = st.session_state.user_profile.get('article_summaries', {}).get(article_url)
        if summary:
            st.write("**Summary:**", summary)

        col1, col2, col3 = st.columns(3)
        if col1.button("💾 Save for Later", key=f"save_{idx}_{user}"):
            st.session_state.user_profile.setdefault('saved_articles', []).append(article_url)
            st.success("Article Saved!")

        feedback_key = f"feedback_{idx}_{user}"
        if feedback_key not in st.session_state.user_profile.get('article_feedback', {}):
            st.session_state.user_profile.setdefault('article_feedback', {})[feedback_key] = None

        feedback = col2.radio("Was this article helpful?", [None, "Yes", "No"], key=feedback_key, horizontal=True)
        if feedback and feedback != st.session_state.user_profile.get('article_feedback', {}).get(feedback_key):
            st.session_state.user_profile.setdefault('article_feedback', {})[feedback_key] = feedback
            st.write(f"Thank you for your feedback!")

        if col3.button("Report", key=f"report_{idx}_{user}"):
            st.write("Thank you for reporting. We will review the article.")

        if article_url:
            st.session_state.user_profile.setdefault('article_read_count', {})[article_url] = st.session_state.user_profile.get('article_read_count', {}).get(article_url, 0) + 1
            read_count = st.session_state.user_profile['article_read_count'][article_url]
            st.write(f"**Read Count:** {read_count}")

            now = datetime.now()
            st.session_state.user_profile.setdefault('read_articles', []).append({'url': article_url, 'timestamp': now, 'user': user})

def generate_wordcloud(articles):
    all_words = " ".join([
        f"{article.get('title', '')} {article.get('description', '')}".strip() 
        for article in articles
    ])
    if all_words:
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords.words('english')).generate(all_words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No articles to generate a word cloud from.")

def main():
    st.set_page_config(page_title="NewsFlow", page_icon="📰", layout="wide")
    st.title("📰 NewsFlow - Your Personalized News Hub")

    # Authentication flow
    if not auth_flow():
        st.sidebar.warning("Please log in or sign up to continue.")
        return

    st.write(f"Welcome, {st.session_state.user}!")

    # Initialize or load user profile
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'selected_categories': ['general'],
            'read_articles': [],
            'preferred_sentiment': 'neutral',
            'saved_articles': [],
            'selected_countries': ['us'],
            'article_feedback': {},
            'article_keywords': {},
            'article_read_count': {},
            'article_summaries': {},
            'language': 'en'
        }

    # Sidebar
    with st.sidebar:
        st.header("Personalization Settings")
        categories = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
        selected_categories = st.multiselect("Preferred Categories", categories, default=st.session_state.user_profile['selected_categories'])
        
        languages = list(LANGUAGES.values())
        current_language_code = st.session_state.user_profile['language']
        current_language_name = LANGUAGES.get(current_language_code, 'english')
        selected_language_name = st.selectbox("Preferred Language", languages, index=languages.index(current_language_name))
        
        sentiment = st.radio("Preferred News Sentiment", ['positive', 'neutral', 'negative'], index=['positive', 'neutral', 'negative'].index(st.session_state.user_profile['preferred_sentiment']))
        countries = st.multiselect("Select Countries", ['us', 'gb', 'in', 'ca', 'au'], default=st.session_state.user_profile['selected_countries'])
        search_query = st.text_input("🔍 Search News by Keywords")

        theme = st.radio("Theme", ["Light", "Dark"], index=0)
        if theme == "Dark":
            st.markdown("""
                <style>
                .stApp {background-color: #1c1c1c; color: white;}
                .stButton>button {background-color: #4CAF50; color: white;}
                </style>
                """, unsafe_allow_html=True)

        if st.button("Update Preferences"):
            selected_language_code = [code for code, name in LANGUAGES.items() if name == selected_language_name][0]
            updated_countries = countries if countries is not None else []
            updated_categories = selected_categories if selected_categories is not None else []
            st.session_state.user_profile.update({
                'selected_categories': updated_categories,
                'preferred_sentiment': sentiment,
                'selected_countries': updated_countries,
                'language': selected_language_code
            })
            st.rerun()

    # Fetch news
    selected_countries = st.session_state.user_profile.get('selected_countries', ['us'])
    selected_categories = st.session_state.user_profile.get('selected_categories', ['general'])
    params = {
        'country': ','.join(selected_countries if selected_countries else ['us']),
        'category': ','.join(selected_categories if selected_categories else ['general']),
        'q': search_query if search_query is not None else '',
        'pageSize': 50
    }
    articles = fetch_news(params)

    if not articles:
        st.warning("No articles found. Try different categories or keywords.")
        return

    # Analyze sentiment and translate if needed
    for article in articles:
        article['sentiment'] = analyze_sentiment(article.get('description', ''))
        if st.session_state.user_profile['language'] != 'en':
            article['title'] = translator.translate(article.get('title', ''), dest=st.session_state.user_profile['language']).text
            article['description'] = translator.translate(article.get('description', ''), dest=st.session_state.user_profile['language']).text

    # Advanced visualizations
    st.subheader("📊 Analytics Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = pd.Series([article['sentiment'] for article in articles]).value_counts()
        fig_sentiment = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, title="News Sentiment Distribution")
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        sources = [article['source']['name'] for article in articles if article['source'].get('name')]
        source_counts = pd.Series(sources).value_counts().head(10)
        fig_geo = px.bar(source_counts, x=source_counts.index, y=source_counts.values, title="Top News Sources")
        st.plotly_chart(fig_geo, use_container_width=True)

    # Trending topics
    st.subheader("🔥 Trending Topics")
    trending = get_trending_topics(articles)
    fig_trending = px.bar(x=[count for word, count in trending], y=[word for word, count in trending], orientation='h', labels={'x': 'Frequency', 'y': 'Keywords'}, title="Trending Topics")
    st.plotly_chart(fig_trending, use_container_width=True)

    # Word Cloud
    st.subheader("☁️ Word Cloud")
    generate_wordcloud(articles)

    # Recommend articles
    recommended_articles = recommend_articles(articles, st.session_state.user_profile.get('read_articles', []))

    # Display articles
    st.subheader("📌 Recommended Articles")
    for idx, (article, _) in enumerate(recommended_articles):
        display_article(article, idx, st.session_state.user)

    # Reading history and saved articles
    with st.expander("📚 Reading History"):
        read_articles = sorted(st.session_state.user_profile.get('read_articles', []), key=lambda x: x['timestamp'], reverse=True)
        for item in read_articles:
            time_str = item['timestamp'].strftime("%Y-%m-d %H:%M:%S")
            st.markdown(f"- [{item['url'][:50]}...]({item['url']}) ({time_str})")

    with st.expander("⭐ Saved Articles"):
        for url in st.session_state.user_profile.get('saved_articles', []):
            st.markdown(f"- [{url[:50]}...]({url})")

    # Export options
    if st.button("Export Reading History to CSV"):
        df = pd.DataFrame(st.session_state.user_profile.get('read_articles', []))
        st.download_button("Download CSV", df.to_csv(), "reading_history.csv", "text/csv")

    if st.button("🔄 Refresh News"):
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Main loop error: {e}")