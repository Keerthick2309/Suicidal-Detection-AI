import streamlit as st
import pandas as pd

st.set_page_config(page_title="Suicide AI", layout="wide")

st.subheader("Evaluation Metrics of All Models")

data = {
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "SVM",
        "LSTM",
        "BILSTM",
        "CNN",
        "BERT"
    ],
    "Accuracy": [0.94, 0.91, 0.94, 0.93, 0.93, 0.93, 0.96],
    "Precision": [0.94, 0.91, 0.94, 0.93, 0.93, 0.93, 0.96],
    "Recall": [0.94, 0.91, 0.94, 0.93, 0.93, 0.93, 0.96],
    "F1 Score": [0.94, 0.91, 0.94, 0.93, 0.93, 0.93, 0.96]
}

df = pd.DataFrame(data)

st.subheader("Comparison Table")
st.dataframe(df, use_container_width=True)

st.subheader("Performance Chart")
st.bar_chart(df.set_index("Model"))

best_model = df.loc[df["F1 Score"].idxmax()]

st.success(f"""
Best Model: **{best_model['Model']}**

- ✔ Highest F1 Score: {best_model['F1 Score']}
- ✔ Accuracy: {best_model['Accuracy']}
""")