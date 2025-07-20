# 🎬 IMDb Movie Review Sentiment Analysis

This project is a sentiment analysis model trained to classify IMDb movie reviews as **positive** or **negative** using deep learning (LSTM with word embeddings). The model processes raw text reviews and predicts the sentiment based on trained patterns in the IMDb dataset.

---

## 🧠 Model Architecture

- **Input**: Text reviews (raw format)
- **Tokenizer**: Keras Tokenizer trained on the dataset
- **Embedding Layer**: Converts words to dense vectors (`input_dim=50000`, `output_dim=128`)
- **LSTM Layer**: Captures long-term dependencies in text (`units=128`)
- **Dense Layer**: Output with sigmoid activation for binary classification

---

## 📌 Features

- Preprocessing of text data (tokenization, padding)
- RNN model built using Keras with:
  - Embedding layer
  - LSTM layer
  - Dense output layer
- Binary classification of movie reviews
- Model saved and loaded via `.h5` file
- Standalone prediction script using `prediction_system.py`

---

## 📁 Dataset

- Source: `IMDb` movie reviews dataset via `tensorflow.keras.datasets.imdb` or custom reviews
- Contains 25,000 training and 25,000 testing examples, labeled as **positive** or **negative**

---

## ⚙️ Installation

```bash
git clone https://github.com/gargdivyansh1/Deep-Learning-Projects/tree/main/IMDb%20Movie%20Reviews
cd imdb-sentiment-analysis
pip install -r requirements.txt
```

## Folder Structure

```bash
IMDb Movie Reviews/
│
├── first.ipynb # Training and evaluation notebook
├── IMDb_rnn.h5 # Trained RNN model
├── prediction_system.py # Python script for making predictions
├── readME.md # Project documentation
└── init.py # Package initializer
```

## Requirements

- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **scikit-learn**
- **pickle**

## 👨‍💻 Developer Info

**Name:** Divyansh Garg  
**Email:** divyanshgarg515@gmail.com   
**GitHub:** [gargdivyansh1](https://github.com/gargdivyansh1)  
**LinkedIn:** [Divyansh Garg](https://www.linkedin.com/in/divyansh-garg515/)
