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

## 📁 Dataset

- Source: `IMDb` movie reviews dataset via `tensorflow.keras.datasets.imdb` or custom reviews
- Contains 25,000 training and 25,000 testing examples, labeled as **positive** or **negative**

---

## ⚙️ Installation

```bash
git clone https://github.com/gargdivyansh1/Deep-Learning-Projects/.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
