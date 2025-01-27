# Text preprocessing and data loading
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower().strip()
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])
