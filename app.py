import streamlit as st
import re
import nltk
import requests
from newspaper import Article
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# --------------------------
# 1. Pre-trained Classification Pipeline
# --------------------------
# For demo purposes, we're using a generic text-classification model (DistilBERT).
# In a real-world project, you'd fine-tune your own model on a Fake News dataset.


# For now use pre trained model, train own model later
model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
classifier = pipeline("text-classification", model=model_name, truncation=True)


def extract_article_content(url):
    """
    Given a URL, scrape and extract the main text of the article.
    Uses newspaper3k for convenience.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        # Maybe log here
        return None

# Clean, tokenize, and remove stop words from raw article text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))

    # Remove non-alphabetic characters, tokenize, and remove stopwords
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens
    processed_text = " ".join(tokens)

    return processed_text

# Use Streamlit for UI
# Scrapes content from article link, preprocesses text, runs through text classification model
def main():
    st.title("Fake News Detector (Demo)")
    st.markdown(
        """
        Demo app
        """
    )

    url = st.text_input("Enter the URL of the news article:")

    if url:
        st.write("**Step 1:** Extracting article text...")
        article_content = extract_article_content(url)

        if not article_content:
            st.error("Failed to scrape the article. Please check the URL or try another.")
            return

        st.write("**Step 2:** Preprocessing the article...")
        clean_text = preprocess_text(article_content)

        # Show small sample of the cleaned text
        if len(clean_text.split()) > 0:
            st.success("Preprocessing complete!")
            st.write("**Sample of processed text:**")
            st.write(" ".join(clean_text.split()[:50]) + " ...")
        else:
            st.error("Article text is empty after preprocessing. Possibly a non-English or invalid article.")
            return

        st.write("**Step 3:** Classifying the article...")

        # Handle longer text by truncating first 512 tokens 
        tokens = clean_text.split()
        truncated_text = " ".join(tokens[:512])

        # Perform classification
        prediction = classifier(truncated_text)[0]
        label = prediction["label"]
        score = prediction["score"]

        if label == "POSITIVE":
            st.success(f"**Predicted Label:** REAL NEWS (Confidence: {score:.2f})")
        else:
            st.warning(f"**Predicted Label:** FAKE NEWS (Confidence: {score:.2f})")

        st.write("**Raw classification result:**", prediction)

        st.write("---")
        st.write("### Explanation")
        st.markdown(
            "explain 1"
            "explain 2"
        )
        if st.button("Run SHAP Explanation (May be slow)"):
            st.info("Running SHAP explanation...")

if __name__ == "__main__":
    main()
