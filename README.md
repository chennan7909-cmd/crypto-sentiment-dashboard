# Crypto Sentiment Dashboard

A real-time dashboard for tracking cryptocurrency market sentiment,
combining price data with sentiment signals to support risk-aware
trading and research decisions.

## Overview

This project monitors sentiment trends for Solana (SOL) alongside
Bitcoin (BTC) price movements, visualizing the relationship between
market mood and price action in an interactive dashboard built with
Streamlit.

## Features

- Real-time SOL/BTC price and sentiment data ingestion
- Interactive time-series visualization
- Sentiment scoring aligned with price trend analysis
- Lightweight deployment via a single app.py entry point

## Tech Stack

Python · Streamlit · Pandas · Matplotlib

## Getting Started

pip install -r requirements.txt
streamlit run app.py

## Data

Raw and processed data files are stored in the /data directory.
Primary dataset: sol_sentiment_price.csv

## Related Projects

- crypto-risk-forecasting: LSTM + XGBoost dual-task prediction pipeline
- sme-credit-risk-pipeline: Real-time credit risk analytics with Kafka
