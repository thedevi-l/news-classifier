import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.set_page_config(page_title="Fake News Classifier", layout="centered")

def train_model():
    real = pd.read_csv("data/real.csv", quotechar='"', sep=",")
    real['label'] = 'REAL'

    fake = pd.read_csv("data/fake.csv", quotechar='"', sep=",")
    fake['label'] = 'FAKE'

    df = pd.concat([real, fake], ignore_index=True)

    df = df.dropna(subset=['text'])

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(C=10, max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy
