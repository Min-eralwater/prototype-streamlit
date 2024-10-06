# TRADEmark: Power News Aggregation and Forecasting Site

This project, TRADEmark, is a proof of concept (PoC) for a multi-functional platform that aggregates financial news, performs price forecasting, executes algorithmic trading strategies, and allows for keyword-based alerts. As currently every aspect is free/open-source; it integrates data from Yahoo Finance, NewsAPI, and utilizes Hugging Face Transformers for text summarization. Additionally, it includes Prophet for time-series forecasting, keyword tracking, and basic mockup functionality for automatic algorithmic trading.

## Features
1. News Aggregation and Summarization:
   * Aggregate news articles from reputable sources using NewsAPI. Users can view up to 10 filtered articles from reputable sources.
   * Filter articles from predefined reputable sources like BBC News, Reuters, etc.
   * Summarize news articles with a transformer model from Hugging Face, which users can use for quick insights
      
2. Forecasting and Anomaly Detection:
   * Historical Brent oil prices are fetched using Yahoo Finance, and then prediction is applied using Facebook Prophet.
   * Adjust forecasts based on user-defined risks associated with key events and keywords.
   * Keyword-based risk factor adjustment using custom weights and risks.
   
4. Real-Time Keyword Alerts:
   * Users can enter keywords (e.g., "oil shortage", "price increase") to track in articles and receive real-time alerts if any match.
   * The app scans news articles for these keywords and triggers alerts.
   * In the future, email functionality to send email alerts when keywords can be added.
   
5. Historical Trend Analysis:
   * Compare the performance of Brent Oil Prices with S&P 500 or other indices.
   * Interactive charts for analyzing trends over the past year.

6. Algorithmic Trading Module:
   * Select a predefined trading strategy: Moving Average Crossover, RSI Strategy, or MACD Strategy.
   * Set strategy parameters and define buy/sell conditions.
   * Simulate or execute mock trades.Review trading results, trade log, and profit/loss summary.

## Tech Stack
* Streamlit: Frontend UI framework for rendering the web application.
* Yahoo Finance (yfinance): Used for fetching historical market data.
* NewsAPI: Fetches the latest news articles related to specific keywords or topics.
* BeautifulSoup: For scraping news content from web pages.
* Hugging Face Transformers: Summarizes news articles using a pre-trained transformer model (bart-large-cnn).
* Prophet: Time-series forecasting for Brent oil price prediction.
* Plotly: For creating interactive plots and charts.
* Pandas: Data manipulation and analysis.

## Next Steps for Future Enhancements
1. Live Trading Integration:
   * Integrate with a real brokerage API (e.g., Alpaca, Interactive Brokers) for live trade execution.
2. Advanced Risk Modeling:
   * Expand risk factor modeling to include refined machine learning-based anomaly detection or external economic indicators.
3. Backtesting Module:
   * Develop a backtesting engine to test strategies against historical data before executing them live.
4. Email Notification:
   * Implement email alerts for keyword-based notifications using smtplib or services like SendGrid for real-time alerts.

## Contact
For any questions, feel free to reach out to the author via email at cheong.minwei@hotmail.co.uk
