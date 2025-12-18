import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
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
