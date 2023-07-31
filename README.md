# Real-Time Twitter Sentiment Analysis

This project uses Machine Learning to perform sentiment analysis on Twitter data in real-time. It's a web application built with Python using the Flask framework.

## Overview

The application fetches Twitter data, preprocesses it, and uses a pre-trained model to predict the sentiment of the tweets. The sentiments are then displayed in a web interface.

**Please note:** Due to recent changes in the Twitter API, fetching of real-time tweets is not possible. Instead, we use the Sentiment140 dataset for training and testing purposes.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Real-Time-Twitter-Sentiment-Analysis.git
cd Real-Time-Twitter-Sentiment-Analysis
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
python3 app.py
```
5. Eventually train the model again:
Download Sentiment140 dataset and put training.1600000.processed.noemoticon.csv in project directory
http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

Run the model training
```bash
python3. preprocessing.py
```
