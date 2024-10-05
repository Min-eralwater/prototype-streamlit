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

st.header("News Aggregation from Multiple Sites")

# API key input
api_key = st.text_input('Enter your NewsAPI key', type='password')

# Query input
query = st.text_input('Enter the topic/keywords you want to search for')

def get_news(api_key, query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    return response.json()

if st.button('Get News'):
    reputable_sources = ['BBC News', 'Financial Times', 'The Guardian', 'Reuters', 'The New York Times']
    if api_key and query:
        news_data = get_news(api_key, query)
        if news_data.get('status') == 'ok':
            articles = news_data.get('articles')
             # Filter articles by reputable sources and limit to 10 articles
            filtered_articles = [article for article in articles if article['source']['name'] in reputable_sources]
            limited_articles = filtered_articles[:10]
            if limited_articles: 
                for article in limited_articles:
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
st.header("Power Price Forecasting and Anomaly Detection", divider=True)

# Define initial system keywords with their default risks and weights
OIL_KEYWORDS = {
    'OPEC': {'default_risk': 'upward', 'weight': 1.5},
    'sanctions': {'default_risk': 'upward', 'weight': 1.2},
    'supply cut': {'default_risk': 'upward', 'weight': 1.3},
    'demand increase': {'default_risk': 'upward', 'weight': 1.1},
    'supply increase': {'default_risk': 'downward', 'weight': 1.0},
    'recession': {'default_risk': 'downward', 'weight': 1.4}
}

# Function to detect keywords and allow users to assign risk factors
def detect_keywords_and_assign_risk(article, keywords):
    detected_keywords = {word: data for word, data in keywords.items() if word in article['content'].lower()}
    user_risk_assignment = {}
    
    st.write(f"Detected factors in the news:")
    
    for keyword, data in detected_keywords.items():
        st.write(f"- {keyword.capitalize()}: Default risk is '{data['default_risk']}' (Weight: {data['weight']})")
        assigned_risk = st.radio(f"Assign risk for {keyword.capitalize()}:", ('upward', 'downward'), index=0 if data['default_risk'] == 'upward' else 1)
        user_risk_assignment[keyword] = {'assigned_risk': assigned_risk, 'weight': data['weight']}
    
    return user_risk_assignment

# Calculate adjustment factor based on user-assigned risks and keyword weights
def calculate_adjustment_factor(risk_assignments):
    adjustment = 0
    for keyword, risk_data in risk_assignments.items():
        risk = risk_data['assigned_risk']
        weight = risk_data['weight']
        if risk == 'upward':
            adjustment += weight
        elif risk == 'downward':
            adjustment -= weight
    return adjustment

# Save user inputs and adjustment factors to a CSV file for future analysis
def save_adjustment_data(risk_assignments, adjustment_factor, filepath="adjustment_data.csv"):
    data = []
    for keyword, risk_data in risk_assignments.items():
        data.append({
            'Keyword': keyword,
            'Assigned Risk': risk_data['assigned_risk'],
            'Weight': risk_data['weight'],
            'Adjustment Factor': adjustment_factor
        })
    df = pd.DataFrame(data)
    
    # Append to CSV file, creating the file if it doesn't exist
    df.to_csv(filepath, mode='a', header=not pd.io.common.file_exists(filepath), index=False)

# Fetch historical data for oil prices
def get_historical_data(ticker="BZ=F", period="1y"):
    data = yf.download(ticker, period=period)
    return data

# Function to get user-defined keywords, risks, and weights
def get_user_defined_keywords():
    st.subheader("Add Custom Keywords, Risks, and Weights")

    custom_keywords = {}

    # Streamlit form to input custom keywords, risks, and weights
    with st.form("custom_keywords_form"):
        custom_keyword = st.text_input("Enter custom keyword")
        custom_risk = st.radio("Assign risk", ("upward", "downward"))
        custom_weight = st.slider("Assign weight", min_value=0.1, max_value=5.0, step=0.1)
        submitted = st.form_submit_button("Add Keyword")
        
        if submitted and custom_keyword:
            custom_keywords[custom_keyword.lower()] = {
                'default_risk': custom_risk,
                'weight': custom_weight
            }
            st.success(f"Added custom keyword: {custom_keyword} with {custom_risk} risk and weight {custom_weight}")

    return custom_keywords

# Fetching Brent Oil data
oil_data = get_historical_data()
#st.line_chart(oil_data['Close'], width=700)

st.subheader("Brent Oil Price Forecasting with Prophet", divider=True)

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

# Example article (this would normally be fetched dynamically from the news API)
sample_article = {
    'content': 'OPEC is cutting supply and demand is increasing due to the upcoming winter season.'
}

# Detect keywords and allow user to assign risk factors
risk_assignments = detect_keywords_and_assign_risk(sample_article, OIL_KEYWORDS)

# Allow user to add their own custom keywords, risks, and weights
custom_keywords = get_user_defined_keywords()

# Combine predefined keywords with user-defined ones
combined_keywords = {**OIL_KEYWORDS, **custom_keywords}

# Detect both predefined and custom keywords in the article
combined_risk_assignments = detect_keywords_and_assign_risk(sample_article, combined_keywords)

# Calculate adjustment factor based on combined user input
adjustment_factor = calculate_adjustment_factor(combined_risk_assignments)
st.write(f"Total Adjustment Factor: {adjustment_factor}")

# Save the risk assignments and adjustment factor for future analysis
save_adjustment_data(combined_risk_assignments, adjustment_factor)

# Apply the adjustment factor to the forecast (1 adjustment unit = 1% change)
adjustment_percentage = adjustment_factor * 0.01
forecast['yhat_adjusted'] = forecast['yhat'] * (1 + adjustment_percentage)

# Plot the adjusted forecast
st.write("Forecast with Adjustment:")
fig_adjusted_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_adjusted_forecast)

st.write(forecast[['ds', 'yhat', 'yhat_adjusted', 'yhat_lower', 'yhat_upper']].tail())

# Section 3: Real-time Alerts with Keyword Tracking
st.subheader("Real-Time Alerts with Keyword Tracking",  divider=True)
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
st.subheader("Historical Trend Analysis",  divider=True)

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

# Section: Algorithmic Trading Module Mockup
st.subheader("Algorithmic Trading Module", divider=True)

# Step 1: Choose a trading strategy
st.write("### Select Trading Strategy")
strategy = st.selectbox(
    "Choose the trading strategy:",
    ("Moving Average Crossover", "RSI Strategy", "MACD Strategy")
)

# Step 2: Set parameters for the strategy
st.write("### Set Strategy Parameters")

if strategy == "Moving Average Crossover":
    short_ma_period = st.number_input("Short Moving Average Period", min_value=1, max_value=100, value=10)
    long_ma_period = st.number_input("Long Moving Average Period", min_value=1, max_value=200, value=50)
elif strategy == "RSI Strategy":
    rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=14)
    rsi_overbought = st.number_input("Overbought RSI Level", min_value=50, max_value=100, value=70)
    rsi_oversold = st.number_input("Oversold RSI Level", min_value=0, max_value=50, value=30)
elif strategy == "MACD Strategy":
    macd_fast = st.number_input("MACD Fast Period", min_value=1, max_value=50, value=12)
    macd_slow = st.number_input("MACD Slow Period", min_value=1, max_value=100, value=26)
    macd_signal = st.number_input("MACD Signal Period", min_value=1, max_value=50, value=9)

# Step 3: Set buy/sell conditions
st.write("### Set Buy/Sell Conditions")
buy_condition = st.text_input("Enter Buy Condition (e.g., 'Close > MA')", "Close > Short MA")
sell_condition = st.text_input("Enter Sell Condition (e.g., 'Close < MA')", "Close < Long MA")

# Step 4: Execute or Simulate Trades
st.write("### Execute or Simulate Trades")
trade_action = st.radio("Action", ("Simulate", "Execute"))

# Example of a trade log
trade_log = pd.DataFrame({
    'Date': ["2024-01-01", "2024-01-02", "2024-01-03"],
    'Action': ["Buy", "Sell", "Buy"],
    'Price': [100.5, 101.7, 102.3],
    'Profit/Loss': [0, 1.2, 0.6]
})

if st.button(f"{trade_action} Trades"):
    if trade_action == "Simulate":
        st.write("Simulation of trades based on the selected strategy and conditions:")
    else:
        st.write("Executing live trades based on the selected strategy and conditions:")
    
    # Show mock trade log
    st.dataframe(trade_log)

# Step 5: Show trade results and summary
st.write("### Trade Summary")
total_trades = len(trade_log)
total_profit_loss = trade_log['Profit/Loss'].sum()

st.write(f"Total Trades Executed: {total_trades}")
st.write(f"Total Profit/Loss: ${total_profit_loss:.2f}")

