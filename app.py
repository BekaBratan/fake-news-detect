import streamlit as st
import joblib
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Load model and metrics
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

with open("metrics.json", "r") as f:
    metrics = json.load(f)

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")

# Display model metrics in a nice layout
st.subheader("📊 Model Metrics")
cols = st.columns(len(metrics))
for i, (key, value) in enumerate(metrics.items()):
    cols[i].metric(label=key, value=f"{value:.4f}")

# Input text for prediction
st.subheader("📝 Enter a news article for prediction")
text_input = st.text_area("Paste the news article here:")

if st.button("Check"):
    if text_input.strip():
        # Predict probability
        proba = model.predict_proba([text_input])[0]

        # Ensure correct mapping to classes
        classes = list(model.classes_)
        real_idx = classes.index(0)
        fake_idx = classes.index(1)

        real_prob = proba[real_idx] * 100
        fake_prob = proba[fake_idx] * 100

        # Determine label
        label = "Real 🟢" if real_prob >= fake_prob else "Fake 🔴"

        st.success(f"Prediction: {label}")
        st.subheader("Confidence:")
        st.progress(int(real_prob))
        st.write(f"Real: {real_prob:.2f}%")
        st.write(f"Fake: {fake_prob:.2f}%")
    else:
        st.warning("Please enter a news article!")
