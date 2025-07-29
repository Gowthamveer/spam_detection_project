# spam_detector.py

# =======================
# 1. Imports
# =======================
import pandas as pd
import string
import pickle
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =======================
# 2. Download NLTK Data
# =======================
nltk.download('stopwords')
nltk.download('wordnet')

# =======================
# 3. Load Dataset
# =======================
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =======================
# 4. Preprocessing Function
# =======================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_message'] = df['message'].apply(preprocess_text)

# =======================
# 5. TF-IDF Vectorization
# =======================
X = df['cleaned_message']
y = df['label']

tfidf = TfidfVectorizer(max_features=3000)
X_vect = tfidf.fit_transform(X)

os.makedirs("vectorizer", exist_ok=True)
with open("vectorizer/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# =======================
# 6. Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# =======================
# 7. Train XGBoost Model
# =======================
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

# =======================
# 8. Evaluate Model
# =======================
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("✅ F1 Score:", round(f1_score(y_test, y_pred), 4))
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))
