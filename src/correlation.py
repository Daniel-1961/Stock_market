import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
analyzer=SentimentIntensityAnalyzer()

# Load datasets

news_df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\raw_analyst_ratings.csv\\raw_analyst_ratings.csv") 
stock_df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\AAPL_historical_data.csv") 
print(news_df.describe())
print(stock_df.describe())

# Convert 'date' columns to datetime
news_df['date'] = pd.to_datetime(news_df['date'])
stock_df['date'] = pd.to_datetime(stock_df['date'])

#  Perform Sentiment Analysis on Headlines
def get_sentiment(text):
        sentiment_score = analyzer.polarity_scores(text)['compound']
        if sentiment_score >= 0.05:
            return 'Positive'
        elif sentiment_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

     
news_df['sentiment'] = news_df['headline'].apply(get_sentiment)

print(news_df[['headline', 'sentiment']].head())

# Aggregate Sentiments by Date and Stock

average_sentiment = news_df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()

#  Calculate Daily Stock Returns

stock_df['daily_return'] = stock_df.groupby('stock')['close'].pct_change()

 #Merge Sentiment Data with Stock Returns

merged_df = pd.merge(average_sentiment, stock_df, on=['date', 'stock'])

# Step 5: Correlation Analysis

def calculate_correlation(df):
    
    sentiment = df['sentiment']
    returns = df['daily_return']
    correlation, p_value = pearsonr(sentiment.dropna(), returns.dropna())
    return correlation, p_value

# Group by stock to calculate correlations for each stock

results = []
for stock, group in merged_df.groupby('stock'):
    corr, p_val = calculate_correlation(group)
    results.append({'stock': stock, 'correlation': corr, 'p_value': p_val})

results_df = pd.DataFrame(results)

# Display correlation results

print("Correlation Results:")
print(results_df)

# Step 6: Visualization

plt.figure(figsize=(10, 6))
plt.bar(results_df['stock'], results_df['correlation'], color='blue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Correlation between Sentiment and Stock Returns by Stock')
plt.xlabel('Stock')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.show()

# Save results
results_df.to_csv('correlation_results.csv', index=False)
