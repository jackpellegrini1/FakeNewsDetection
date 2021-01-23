import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
import time
import string

# import data
fake_raw_data = pd.read_csv("FakeNewsDetection/data/Fake.csv")
real_raw_data = pd.read_csv("FakeNewsDetection/data/True.csv")

# preprocessing
length_fake = len(fake_raw_data)
length_real = len(real_raw_data)

fake = pd.DataFrame({'text': fake_raw_data['text'], 'class': [0]*length_fake})
real = pd.DataFrame({'text': real_raw_data['text'], 'class': [1]*length_real})

data = fake.append(real, ignore_index=True)

def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

data["text"] = data.text.map(remove_punctuation)

stop = set(stopwords.words("english"))
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)
data["text"] = data.text.map(remove_stopwords)

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

counter = counter_word(data.text)

num_unique_words = len(counter)

X_train, X_test, Y_train, Y_test = train_test_split(data["text"], data["class"], test_size=0.2)

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# This was for trying to find the right size of the padding

# train_lengths = []

# for i in range(35918):
#     train_lengths.append(len(X_train[i]))

# test_lengths = []

# for i in range(8980):
#     test_lengths.append(len(X_test[i]))

# print(max(train_lengths))
# print(max(test_lengths))

max_length = 700

X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

print(X_train.shape)

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))

model.add(layers.LSTM(62, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test), verbose=2)