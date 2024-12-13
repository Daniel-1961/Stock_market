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
