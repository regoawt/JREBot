from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow.keras.utils as ku
import tensorflow as tf
import tensorflow_cloud as tfc
import numpy as np
import random


tokenizer = Tokenizer()
data = open('NLP/corpus.txt').read()

corpus = data.lower().split('\n ')
tokenizer.fit_on_texts(corpus)
num_of_unique_words = len(tokenizer.word_index) + 1
print('Number of unique words: '+ str(num_of_unique_words))


# In[ ]:


n = 6
input_texts = []
for i in range(len(corpus)):
    sentence_list = corpus[i].rstrip(' ').split(' ')
    if len(sentence_list) > n:
        startpoint = len(sentence_list)-(n-1)
        endpoint = n-3
    else:
        startpoint = 1
        endpoint = len(sentence_list)-1

    for j in range(startpoint):
        for k in range(endpoint):
            line = sentence_list[j+k:j+n]
            input_texts.append(line)
#print(input_texts[:20])
random.shuffle(input_texts)
print('Number of samples: '+str(len(input_texts)))


# In[ ]:


input_sequences = tokenizer.texts_to_sequences(input_texts)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=n, padding='pre'))

predictors = input_sequences[:,:-1]
labels = input_sequences[:,-1]


# In[ ]:


class mycallback(Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get('accuracy') > 0.8:
            print('Stopping early')
            self.model.stop_training = True

callback1 = mycallback()

callback2 = EarlyStopping(monitor='accuracy', min_delta = 0.002)


# In[ ]:


model = Sequential([
    Embedding(num_of_unique_words, 100, input_length=n-1),
    Bidirectional(LSTM(1024, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(512),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_of_unique_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(num_of_unique_words, activation='softmax')
])
adam = Adam(learning_rate=0.001, decay=1e-5)
sgd = SGD(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())


# In[ ]:


history = model.fit(predictors,labels, epochs=150, callbacks=[callback1, callback2],
                    batch_size=200, validation_split=0.15)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training loss')
plt.legend()

plt.show()


# In[30]:


model.save('tmp/jrebot', overwrite=True)


# In[36]:


seed = "the world is going"

next_words = 50
phrase = []
def randomize_word(sequence, temperature=1):
    probs = model.predict(sequence)
    rescaled_logits = tf.math.log(probs)/temperature
    word_id = tf.random.categorical(rescaled_logits, num_samples=1)

    return word_id


for _ in range(next_words):
    phrase = seed.split(' ')[-(n-1)::]
    sequence = tokenizer.texts_to_sequences([phrase])
    sequence = pad_sequences(sequence, maxlen=n-1, padding='pre')
    #predicted = model.predict_classes(sequence)
    predicted = randomize_word(sequence,temperature=0.5)
    next_word = ''
    for word,index in tokenizer.word_index.items():
        if index == predicted[0]:
            next_word = word
            break
    seed += ' ' + next_word

print(seed)



# In[ ]:
