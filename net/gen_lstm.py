import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import json
import sys
import os


if os.environ['CUDA_VISIBLE_DEVICES'] == '':
	exit()

def to_onehot(i, v):
	x= np.zeros(v)
	x[i] = 1
	return x


def from_onehot(x):
	return x.argmax()


def vec2word(x, rw_index):
	i = from_onehot(x)
	return rw_index[i]


def word2vec(w, w_index):
	x = w_index[w]
	return to_onehot(x)


def create(v, max_len):
	model = Sequential()
	model.add(LSTM(128, input_shape=(maxlen, v))
	model.add(LSTM(128))
	model.add(Dense(v, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimzer=RMSprop(lr=0.01))
	return model

corpus_0 = pd.read_csv("../data/polygon.txt", sep="\n")
corpus_0 = corpus_0['title']

corpus_1 = pd.read_csv("../data/abcnews.txt", sep="\n")
corpus_1 = corpus_1['title']

corpus_2 = pd.read_csv("../data/nytimes.txt", sep="\n")
corpus_2 = corpus_2['title']

corpus = pd.concat([corpus_0, corpus_1, corpus_2])

#TODO get word_index
json_0 = open('dict_poly.json').read()
word_index = json.loads(json_0)
#TODO get rword_index
json_1 = open('rdict.json').read()
rword_index = json.loads(json_1)

max_len = 3
step = 1
x = []
y = []

for sequence in tokenize it lol corpus:
	for i, s in enumerate(sequence):	
		x.append(sequence[i: i + maxlen])
		y.append(sequence[i + maxlen])
























