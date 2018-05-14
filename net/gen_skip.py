import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import skipgrams, make_sampling_table

import pandas as pd
import os

if os.environ['CUDA_VISIBLE_DEVICES'] == '':
	exit()

def to_onehot(i, v):
	x = np.zeros(v)
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


def create():
	embedding_dim = 128
	model = Sequential()
	model.add(Dense(embedding_dim, activation='relu', input_shape=(vocab_size,)))
	model.add(Dropout(0.2))
	model.add(Dense(vocab_size, activation='softmax'))

	model.compile(optimizer='adam', loss='MSE')
	return model


def train(model, epoch):
	for _ in range(epoch):
		loss = 0.0
		for i, line in enumerate(tokenizer.texts_to_sequences(corpus)):
			if len(line) < 5:
				continue
			sequence, label = skipgrams(sequence=line, vocabulary_size=vocab_size, window_size=2, negative_samples=0.0)
			x = []
			y = []
			for j, s in enumerate(sequence):
				x.append(to_onehot(s[0], vocab_size))
				y.append(to_onehot(s[1], vocab_size))
				
			x = np.array(x)
			y = np.array(y)
			loss += model.train_on_batch(x, y)
		print(loss)
	return loss


corpus_0 = pd.read_csv("../data/polygon.txt", sep="\n")
corpus_0 = corpus_0[corpus_0.title.str.contains("Not found") == False]
corpus_0 = corpus_0['title']

corpus_1 = pd.read_csv("../data/abcnews.txt", sep="\n")
corpus_1 = corpus_1[corpus_1.title.str.contains("ABC News - Breaking News") == False]
corpus_1 = corpus_1['title']

corpus_2 = pd.read_csv("../data/nytimes.txt", sep="\n")
corpus_2 = corpus_2[corpus_2.title.str.contains("Page Not Found") == False]
corpus_2 = corpus_2['title']

corpus = pd.concat([corpus_0, corpus_1, corpus_2])


vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, filters='"#%&()*+,.-;:<=>@[\\]`\'{|}~\t\n', split=" ", lower=True)
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
print(word_index)
print("vocabulary size: ", vocab_size)
reverse_word_index = {v: k for k, v in word_index.items()}




model = create()
loss = train(model, 20)
print(loss)
model.save('anp_20.h5')
