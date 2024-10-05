import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf
from transformers import pipeline
from prophet import Prophet 
from prophet.plot import plot_plotly
import plotly.graph_objects as go


# Section 1: News Aggregation and Summarization using Hugging Face Transformers
st.title("TRADEmark: Power News Aggregation and Forecasting Site")

st.subheader("News Aggregation from multiple Sites")

# API key input
api_key = st.text_input('Enter your NewsAPI key', type='password')

# Query input
query = st.text_input('Enter the topic/keywords you want to search for')

def get_news(api_key, query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    return response.json()

if st.button('Get News'):
        if api_key and query:
            news_data = get_news(api_key, query)
            if news_data.get('status') == 'ok':
                articles = news_data.get('articles')
                for article in articles:
                    st.subheader(article['title'])
                    st.write(article['description'])
                    st.write(f"Source: {article['source']['name']}")
                    st.write(f"Published At: {article['publishedAt']}")
                    st.write(f"[Read more]({article['url']})")
                    st.write("---")
            else:
                st.error("Failed to fetch news articles. Please check your API key and query.")
        else:
            st.error("Please try again later.")

st.write("Summarize news articles from a predefined list of URLs or enter your own.")

# List of predefined URLs (you can modify or add more)
predefined_urls = {
    "CNA Energy Article": "https://www.channelnewsasia.com/business/energy-firms-boost-gas-exploration-southeast-asia-meet-growing-demand-4163071",
    "Offshore Technology Article":"https://www.offshore-technology.com/news/uk-north-sea-oil-and-gas-industry-failing-to-shift-investments-to-renewable-energy/?cf-view",
    "Reuters Energy Article": "https://www.reuters.com/markets/deals/ftc-allows-chevron-hess-deal-bars-john-hess-board-2024-09-30/"
}

# User can select a predefined URL or input their own
option = st.selectbox(
    "Select a news source or enter a URL:",
    options=["Select predefined URL", "Enter your own URL"]
)

if option == "Select predefined URL":
    url = st.selectbox("Choose a predefined URL:", options=list(predefined_urls.keys()))
    selected_url = predefined_urls[url]  # Get the selected URL
else:
    selected_url = st.text_input("Enter a URL to summarize:")

if selected_url:
    st.write(f"Fetching and summarizing news from: {selected_url}")
    @st.cache_data(ttl=3600)  # Cache for 1 hour to avoid frequent fetching
    def fetch_news(url):
        """Fetch news articles or text from a URL."""
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            page = requests.get(url, headers=headers)
            soup = BeautifulSoup(page.content, "html.parser")

            # Extract article content (this depends on the website structure, simplified here)
            paragraphs = [p.get_text() for p in soup.find_all('p', limit=5)]
            return " ".join(paragraphs)
        except Exception as e:
            return f"Error fetching the URL: {e}"

    article_content = fetch_news(selected_url)

    if article_content:
        st.write(f"**Original Content from URL**")
        st.write(article_content)

        # Caching the summarization model
        @st.cache_resource
        def load_summarizer():
            return pipeline("summarization", model="facebook/bart-large-cnn")
        # Summarization using Hugging Face transformers
        st.subheader("Summarized News with Hugging Face")

        # Load the summarizer model
        summarizer = load_summarizer()

        def summarize_news_huggingface(news_text):
            """Summarizes news articles using Hugging Face"""
            summary = summarizer(news_text, max_length=100, min_length=25, do_sample=False)
            return summary[0]['summary_text']

        try:
            summary = summarize_news_huggingface(article_content)
            st.write(f"**Summarized Content:** {summary}")
        except Exception as e:
            st.write(f"- Error summarizing article: {e}")
    else:
        st.write("No content fetched from the URL")

# Section 2: Forecasting and Anomaly Detection for Power Prices
st.subheader("Power Price Forecasting and Anomaly Detection")

@st.cache_data(ttl=86400) # Cache historical data for 1 day
def get_historical_data(ticker="BZ=F", period="1y"):
    """Fetch historical data for Brent crude prices from Yahoo Finance"""
    data = yf.download(ticker, period=period)
    return data

# Fetching Brent Oil data
oil_data = get_historical_data()
st.line_chart(oil_data['Close'], width=700)

# Prophet Forecasting for Brent Oil Prices
st.subheader("Brent Oil Price Forecasting with Prophet")

# Prepare data for Prophet (must have 'ds' for date and 'y' for values)
@st.cache_data(ttl=86400)  # Cache Prophet results for 1 day
def fit_prophet_model(data):
    df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    m = model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return m, forecast

m, forecast = fit_prophet_model(oil_data)

# Plotting the forecast
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast)

st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Section 3: Real-time Alerts with Keyword Tracking
st.subheader("Real-Time Alerts with Keyword Tracking")
st.write("Set up alerts for specific keywords (e.g., 'price increase', 'oil shortage')")

keywords = st.text_input("Enter keywords to track for alerts", "oil, power price")
keywords = [kw.strip() for kw in keywords.split(",")]

def keyword_tracking(articles, keywords):
    """Track if any keywords are found in the articles"""
    alerts = []
    for article in articles:
        for keyword in keywords:
            if keyword.lower() in article.lower():
                alerts.append(f"Alert: {keyword} found in article: {article}")
    return alerts

# Running keyword tracking on fetched articles
if selected_url:
    st.subheader("Keyword Alerts")
    alerts = keyword_tracking([article_content], keywords)
    for alert in alerts:
        st.write(alert)

# Section 4: Historical Trend Analysis
st.subheader("Historical Trend Analysis")

@st.cache_data(ttl=86400)
def compare_with_index(oil_data, index_ticker="^GSPC"):
    """Compare Brent oil prices with another index (e.g., S&P 500)"""
    index_data = yf.download(index_ticker, period="1y")
    combined_data = pd.DataFrame({
        "Oil Prices": oil_data['Close'],
        "Index": index_data['Close']
    })
    combined_data.dropna(inplace=True)
    return combined_data

# Fetch and compare with S&P 500 index
comparison_data = compare_with_index(oil_data)
st.line_chart(comparison_data)

st.write("You can now analyze trends between Brent Oil and S&P 500 index.")
