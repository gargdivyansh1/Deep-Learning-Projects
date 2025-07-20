import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
index_offset = 3
word_index = {k: (v + index_offset) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

model = load_model(r'C:\Users\divya\OneDrive\Desktop\Machine Learning\Deep Learning Projects\IMDb Movie Reviews\IMDb_rnn.h5')

def review_to_sequence(review, vocab_size = 50000, sequence_length = 200):
    tokens = review.lower().split()
    sequence = []

    for word in tokens :
        index = word_index.get(word, 2)
        if index < vocab_size:
            sequence.append(index)

    padded = pad_sequences([sequence], maxlen = sequence_length)
    return padded 

def predict_sentiment(review_text):
    sequence = review_to_sequence(review_text)
    prediction = model.predict(sequence)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive"
    else :
        sentiment = 'Negative'

    print(f'\nReview: {review_text}\nPrediction Score: {prediction:-4f}\nSentiment: {sentiment}')
    return sentiment

if __name__ == '__main__':
    review_input = input("Enter a movie review: ")
    predict_sentiment(review_input)