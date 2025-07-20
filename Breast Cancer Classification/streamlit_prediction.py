import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Title
st.title("🔬 Breast Cancer Prediction using Neural Network")

# Load data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['label'] = data.target
    return df, data.feature_names

df, features = load_data()

# Train/test split
X = df.drop(columns='label', axis=1)
Y = df['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the model
model = keras.models.load_model('brest_cancer_model.h5')

# Input form
st.header("📋 Enter Feature Values")

input_values = []
for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, value=float(df[feature].mean()), format="%.4f")
    input_values.append(value)

if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    if predicted_class == 1:
        st.success(f"🟢 The tumor is predicted to be **Benign** with confidence {confidence:.2f}")
    else:
        st.error(f"🔴 The tumor is predicted to be **Malignant** with confidence {confidence:.2f}")

# Show dataset and description
with st.expander("📊 Show Sample Data"):
    st.write(df.head())

with st.expander("ℹ️ About this app"):
    st.markdown("""
        This app uses a Neural Network to predict whether a breast tumor is benign or malignant 
        based on 30 features from the Breast Cancer Wisconsin dataset.
    """)
