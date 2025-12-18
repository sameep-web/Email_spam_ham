import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

nltk.data.path.append("/home/adminuser/nltk_data")

nltk.download("punkt", download_dir="/home/adminuser/nltk_data")
nltk.download("stopwords", download_dir="/home/adminuser/nltk_data")
nltk.download("wordnet", download_dir="/home/adminuser/nltk_data")

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

st.title("📧 Spam vs Ham Classifier")
st.write("Enter an email message and find out whether it is **SPAM** or **HAM**.")

user_input = st.text_area("✍️ Email Text")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some email content.")
    else:
        cleaned = preprocess(user_input)
        vect_input = vectorizer.transform([cleaned])
        prediction = model.predict(vect_input)[0]

        if prediction == 1:
            st.error("🚫 This email is classified as **SPAM**.")
        else:
            st.success("This email is classified as **HAM**.")
