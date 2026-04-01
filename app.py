import streamlit as st

st.set_page_config(page_title="Suicide AI", layout="wide")
st.title("AI-Based Suicide Detection System")

st.markdown("""
## Project Overview
This application detects suicidal intent from social media text using Artificial Intelligence.

It uses multiple models:
- Machine Learning (Naive Bayes, Logistic Regression, SVM)
- Deep Learning (LSTM, BiLSTM,, CNN)
- Transformer (BERT)

---

## Objective
To automatically identify suicidal thoughts in text data and help in early mental health intervention.

---

## How It Works
1. Enter any text (like a social media post)
2. Choose a model
3. Get prediction instantly

---

## Output
The system classifies text into:
- **Suicidal**
- **Non-Suicidal**

---

## Features
- Multi-model prediction
- Model comparison
- Simple UI for testing

---

Go to the **Prediction page** from the sidebar to test the model.
""")
