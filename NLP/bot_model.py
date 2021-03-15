from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np

tokenizer = Tokenizer()
data = open('Joe-Rogan-Gets-High-And-Rants-About-Stuff-_320-kbps_.flac.txt').read()

corpus = data.lower().split(' ')
tokenizer.fit_on_texts(corpus)
num_of_unique_words = len(tokenizer.word_index) + 1

n = 5
input_texts = []
for i in range(len(corpus)-n):
    line = corpus[i:i+5]
    input_texts.append(line)

input_sequences = tokenizer.texts_to_sequences(input_texts)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=n, padding='post'))

predictors = input_sequences[:,:-1]
labels = input_sequences[:,-1]

labels = ku.to_categorical(labels,num_classes=num_of_unique_words)

model = Sequential([
    Embedding(num_of_unique_words, 100, input_length=n-1),
    Bidirectional(LSTM(150, return_sequences=True)),
    Dropout(0.3),
    LSTM(100),
    Dense(num_of_unique_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(num_of_unique_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
