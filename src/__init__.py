import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
analyzer=SentimentIntensityAnalyzer()

df=pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\raw_analyst_ratings.csv\\raw_analyst_ratings.csv")
print(df)
def headline_length():
    df['head_lenght']=df['headline'].apply(len)
    print(df['head_lenght'])
    headline_status=df['head_lenght'].describe()
    print(headline_status)
def article_per_publisher():
    publisher_counts=df['publisher'].value_counts()
    top_publishers=publisher_counts.head(10)
    print(top_publishers)



def time_analysis():
    df['date']=pd.to_datetime(df['date'],format='mixed', errors='coerce',utc=True)
    print(df['date'])
    #EXRACTING COMPONENTS FOR ANALYSIS BY WEAK, MONTH AND YEAR

    df['day_of_weak']=df['date'].dt.day_name()
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year

    #articles publishe by days of the weak, month,year
    day_counts=df['day_of_weak'].value_counts()
    month_counts=df['month'].value_counts()
    year_counts=df['year'].value_counts()

    # articles published by days of the weak

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=df['day_of_weak'], order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Articles Published by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Articles')
    plt.show()
    #################ARTICLES PUBLISHED BY MONTS OF THE YEAR ###################################
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=df['month'])
    plt.title('Number of Articles Published by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Articles')
    plt.show()
#ARTICLES DISTIRBUTION BY YEAR
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=df['year'])
    plt.title('Number of Articles Published by year')
    plt.xlabel('years')
    plt.ylabel('Number of Articles')
    plt.show()




#SENTIMENTAL ANALYSIS


    def get_sentiment(text):
        sentiment_score = analyzer.polarity_scores(text)['compound']
        if sentiment_score >= 0.05:
            return 'Positive'
        elif sentiment_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['headline'].apply(get_sentiment)

    #Check the results
    print(df[['headline', 'sentiment']].head())


#SENTIMENT ANALAYSIS GRAPH


    plt.figure(figsize=(8, 6))
    sns.countplot(x=df['sentiment'], data=df)
    plt.title('Sentiment Distribution of Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

#TIME SERIES ANALYSIS using daily article
def time_series():

    daily_articles = df.groupby(df['date'].dt.date).size()

    plt.figure(figsize=(12, 6))
    daily_articles.plot(kind='line', color='blue', marker='o')
    plt.title('Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()

# TIME SERIES BY DAY


    df['hour'] = df['date'].dt.hour

    
    hourly_articles = df['hour'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    hourly_articles.plot(kind='bar', color='orange')
    plt.title('Publishing Times Distribution')
    plt.xlabel('Hour of Day (24-hour format)')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()



#publisher analysis
def publisher_analysis():

    publisher_counts = df['publisher'].value_counts()

    #Display top publishers
    print("Top Publishers Contributing to the News Feed:")
    print(publisher_counts.head(10))

    #Plot the top publishers

    plt.figure(figsize=(12, 6))
    publisher_counts.head(10).plot(kind='bar', color='skyblue')
    plt.title('Top Publishers by Number of Articles')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()


# # Extract domain names from publisher emails

def domain_Name():
    df['domain'] = df['publisher'].apply(lambda x: re.search(r'@([\w.]+)', x).group(1) if '@' in x else None)

    # Count the number of articles per domain
    domain_counts = df['domain'].value_counts()

    # Display top domains
    print("Top Domains Contributing to the News Feed:")
    print(domain_counts.head(10))

    # Plot the top domains
    plt.figure(figsize=(12, 10))
    domain_counts.head(10).plot(kind='bar', color='orange')
    plt.title('Top Domains by Number of Articles')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

headline_length()
time_analysis()
time_series()
publisher_analysis()
domain_Name()
