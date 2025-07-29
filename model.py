import re
import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('spam_ham_dataset.csv')
df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
df.drop_duplicates(inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, min_df=5, max_df=0.9, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'),
    "SVM": LinearSVC(C=1.0, max_iter=1000)
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    if acc > best_score:
        best_model = model
        best_score = acc
        best_name = name

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\nBest model: {best_name} saved with accuracy: {best_score:.4f}")
