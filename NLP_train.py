import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LSTM
import time
from keras import utils as np_utils
training = np.genfromtxt('data.csv', delimiter=',', skip_header=1, usecols=(1, 2), dtype=None,encoding='ISO-8859-1')
train_x = [x[1] for x in training]
train_y = np.asarray([x[0] for x in training])
max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)
dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
for text in train_x:
    wordInfices = convert_text_to_index_array(text)
    allWordIndices.append(wordInfices)
allWordIndices = np.asarray(allWordIndices)
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
train_y = keras.utils.np_utils.to_categorical(train_y, 2)
model = Sequential()
model.add(Dense(512,input_shape=(max_words,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
t = time.time()
print(time.ctime(t))
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.1, shuffle=True)
t2 = time.time()
print(time.ctime(t2))
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
print('model saved')
