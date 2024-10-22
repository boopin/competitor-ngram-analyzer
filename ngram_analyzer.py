import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Stopwords list
stop_words = set(stopwords.words('english'))

# Display NumPy version to ensure it's properly installed
st.write(f"NumPy version: {np.__version__}")

# Function to extract page content
def get_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

# Function to clean text and remove stopwords
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Function to generate n-grams
def generate_ngrams(tokens, n):
    n_grams = ngrams(tokens, n)
    return Counter(n_grams)

# Function to generate word cloud from tokens and clear existing plots
def generate_wordcloud(tokens):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
    
    # Clear previous plot to prevent overlap
    plt.clf()
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Function to convert n-grams to CSV format and download
def download_ngrams_as_csv(ngrams, filename="ngrams.csv"):
    df = pd.DataFrame(ngrams.most_common(10), columns=["N-gram", "Count"])
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download N-gram data as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# Streamlit UI
st.title("Advanced N-gram Analyzer for Competitor URLs")

# Input for Competitor URLs
url1 = st.text_input("Enter Competitor URL 1", "")
url2 = st.text_input("Enter Competitor URL 2 (optiona
