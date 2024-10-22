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
url2 = st.text_input("Enter Competitor URL 2 (optional)", "")

# Process for URL 1
if url1:
    text_content1 = get_page_content(url1)
    st.write("Extracted content from URL 1 (first 500 characters): ", text_content1[:500])

    tokens1 = clean_text(text_content1)
    
    n = st.selectbox("Select N-gram (1 for Unigram, 2 for Bigram, 3 for Trigram)", [1, 2, 3])

    # Generate N-grams for URL 1
    if st.button("Generate N-grams for URL 1"):
        ngram_counts1 = generate_ngrams(tokens1, n)
        st.write(f"Top {n}-grams for URL 1:")
        for ngram, count in ngram_counts1.most_common(10):
            st.write(f"{ngram}: {count}")

        # Add download button for the n-gram results
        download_ngrams_as_csv(ngram_counts1, filename="ngrams_url1.csv")

    # Generate Word Cloud for URL 1
    if st.button("Generate Word Cloud for URL 1"):
        st.write("Word Cloud for URL 1:")
        generate_wordcloud(tokens1)

# Process for URL 2 (if provided)
if url2:
    text_content2 = get_page_content(url2)
    st.write("Extracted content from URL 2 (first 500 characters): ", text_content2[:500])

    tokens2 = clean_text(text_content2)
    
    # Generate N-grams for URL 2
    if st.button("Generate N-grams for URL 2"):
        ngram_counts2 = generate_ngrams(tokens2, n)
        st.write(f"Top {n}-grams for URL 2:")
        for ngram, count in ngram_counts2.most_common(10):
            st.write(f"{ngram}: {count}")

        # Add download button for the n-gram results
        download_ngrams_as_csv(ngram_counts2, filename="ngrams_url2.csv")

    # Generate Word Cloud for URL 2
    if st.button("Generate Word Cloud for URL 2"):
        st.write("Word Cloud for URL 2:")
        generate_wordcloud(tokens2)

# Optional: Add comparison of N-grams between the two URLs
if url1 and url2 and st.button("Compare N-grams for both URLs"):
    st.write(f"Comparison of top {n}-grams between URL 1 and URL 2:")
    common_ngrams = set(ngram_counts1.elements()) & set(ngram_counts2.elements())
    st.write(f"Common N-grams between URL 1 and URL 2: {list(common_ngrams)}")
