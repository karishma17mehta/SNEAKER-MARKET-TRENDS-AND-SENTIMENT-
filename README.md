# Sneaker PostHype Trend Forecasting

This project explores how sneaker trends are evolving in the post-hype era, focusing on the online behavior of Gen Z consumers. Using Google Trends, Reddit scraping, sentiment analysis, and BERT topic modeling, we uncover engagement shifts across major sneaker brands. A Tableau dashboard ties all this together into an interactive visualization.

---

## Dashboard

An interactive Tableau dashboard is built using the combined dataset and includes:
- Google Trends (historical + forecasted)
- Sentiment analysis per brand
- Reddit post volume over time
- Brand trend scorecard

📍 [View it on Tableau Public](#) _(https://public.tableau.com/app/profile/karishma.mehta8733/viz/TABLEAU_DASHBOARD_SNEAKER_MARKET/Dashboard1)_

---

## Key Features

- **Time Series Forecasting** using Facebook Prophet on Google Trends data
- **Reddit Scraping** using PRAW (e.g., r/Sneakers, r/FashionReps)
- **Topic Modeling** with BERTopic for trend themes
- **Sentiment Classification** using RoBERTa (CardiffNLP)
- **Interactive Tableau Dashboard** for storytelling and exploration

---

## Files

| File | Description |
|------|-------------|
| `Sneaker_PostHype_TrendForecasting_1.py` | Full Python workflow for data collection, NLP, sentiment, and forecasting |
| `Final_Forecas_Dataset.csv` | Combined dataset used for Tableau visualization |
| `reddit_filtered_sneaker_posts.csv` | Raw Reddit data collected |
| `sneaker_google_trend_1.csv` | Google Trends data used for forecasting |
| `TABLEAU_DASHBOARD.twbx` | Packaged Tableau workbook (optional) |

---
