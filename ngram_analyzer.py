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

# Function
