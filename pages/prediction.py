import streamlit as st
import torch
import joblib
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

st.set_page_config(page_title="Suicide AI", layout="wide")

@st.cache_resource
def load_model():
    lr_model = joblib.load("models/lr_model.pkl")
    nb_model = joblib.load("models/nb_model.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
    bilstm_model = tf.keras.models.load_model("models/bilstm_model.h5")
    cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
    tokenizer = joblib.load("models/tokenizer.pkl")

    bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
    bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")
    explainer = LimeTextExplainer(class_names=["Non-Suicidal", "Suicidal"])

    return lr_model, nb_model, svm_model, tfidf, lstm_model, bilstm_model, cnn_model, tokenizer, bert_model, bert_tokenizer, explainer

lr_model, nb_model, svm_model, tfidf, lstm_model, bilstm_model, cnn_model, tokenizer, bert_model, bert_tokenizer, explainer = load_model()

def prediction(text, model_name):
    if model_name == "Logistic Regression":
        vec = tfidf.transform([text])
        pred = lr_model.predict(vec)[0]
    elif model_name == "Naive Bayes":
        vec = tfidf.transform([text])
        pred = nb_model.predict(vec)[0]
    elif model_name == "SVM":
        vec = tfidf.transform([text])
        pred = svm_model.predict(vec)[0]
    elif model_name == "LSTM":
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = (lstm_model.predict(padded) > 0.5).astype(int)[0][0]
    elif model_name == "BILSTM":
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = (bilstm_model.predict(padded) > 0.5).astype(int)[0][0]
    elif model_name == "CNN":
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = (cnn_model.predict(padded) > 0.5).astype(int)[0][0]
    elif model_name == "BERT":
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return "Suicidal" if pred == 1 else "Non-Suicidal"

def bert_predict_proba(texts):
    inputs = bert_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

def explain_text(text):
    exp = explainer.explain_instance(
        text,
        bert_predict_proba,
        num_features=6
    )
    return exp


st.title("Suicidal Detection Prediction")

model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Naive Bayes", "SVM", "LSTM", "BILSTM", "CNN", "BERT"]
)

text = st.text_area("Enter Text")
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter Text")
    else:
        result = prediction(text, model_choice)
        st.success(f"{model_choice} Prediction: {result}")
        if model_choice == "BERT":
            st.subheader("Explanation (Important Words)")

            exp = explain_text(text)

            # List view
            st.write(exp.as_list())

            # 🔥 Beautiful HTML highlight
            st.subheader("Visual Explanation")
            st.components.v1.html(exp.as_html(), height=400)