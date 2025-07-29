# streamlit_app.py

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# =======================
# 1. Load Model & Vectorizer
# =======================
with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =======================
# 2. Preprocessing Function
# =======================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# =======================
# 3. Streamlit UI
# =======================
st.title("📩 Spam Message Classifier")
st.markdown("Enter a message and the model will predict if it's **Spam** or **Not Spam**.")

user_input = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    cleaned = preprocess_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.error("🚨 This is **SPAM**.")
    else:
        st.success("✅ This is **NOT Spam**.")
