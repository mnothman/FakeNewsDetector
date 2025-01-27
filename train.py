import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def load_data():
    fake = pd.read_csv("data/News_dataset/Fake.csv")
    true = pd.read_csv("data/News_dataset/True.csv")

    fake['label'] = 0  # Fake news
    true['label'] = 1  # Real news

    # Combine the datasets and then shuffle
    data = pd.concat([fake, true], ignore_index=True)
    data = data.sample(frac=1, random_state=42)

    return data

def preprocess_and_vectorize(data):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    return X, y, vectorizer

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Model trained successfully!")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model and vectorizer
    with open("models/fake_news_model.pkl", "wb") as model_file:
        pickle.dump((model, vectorizer), model_file)

if __name__ == "__main__":
    data = load_data()
    X, y, vectorizer = preprocess_and_vectorize(data)
    train_model(X, y)
