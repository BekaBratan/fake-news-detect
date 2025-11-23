# Fake News Detector

A Python-based machine learning model that detects whether a news article is **real** or **fake**.  
The project uses **text preprocessing**, **TF-IDF vectorization**, and **Logistic Regression** for classification.

---

## Features

- Text preprocessing: lowercasing, punctuation removal, stopword removal, lemmatization  
- TF-IDF vectorization for feature extraction  
- Logistic Regression classifier  
- Supports single news prediction via a simple interface (Streamlit)  
- Model metrics stored for evaluation: Accuracy, Precision, Recall, F1, ROC-AUC  

---

## Model Comparison & Selection

Several models and approaches were evaluated:

- **Logistic Regression** with TF-IDF  
- **Multinomial Naive Bayes** with TF-IDF  
- **SVM** with TF-IDF  
- **Transformer embeddings** (all-MiniLM-L6-v2) with Logistic Regression and SVM  

**Why Logistic Regression + TF-IDF was chosen:**

- Fast inference suitable for real-time predictions  
- High accuracy and stable performance  
- Provides probability/confidence for predictions  
- Simpler and easier to deploy compared to transformer embeddings  

---

## Installation

```bash
git clone https://github.com/yourusername/fake-news-detect.git
cd fake-news-detect
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```
- Paste a news article into the text area
- Click Check to see if it is Real or Fake

---

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Requirements

- Python 3.10+
- scikit-learn
- nltk
- pandas
- streamlit

---

## Files
- Fake_News_Detection.ipynb is Jupyter Notebook that for training and initializing model
- metrics.json is metrics file
- model.pkl is ready trained model
- app.py is file for running and testing model via streamlit
