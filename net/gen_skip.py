from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from pickle import dump
import numpy as np
import pandas as pd
import sys
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


def train(model, tokenizer, v, epoch):
	history = ""
	for _ in range(epoch):
		loss = 0.0
		for i, seq in enumerate(tokenizer.texts_to_sequences(corpus)):
			if len(seq) < 5:
				continue
			pairs, _ = skipgrams(sequence=seq, vocabulary_size=v, window_size=2, negative_samples=0.0)
			x = []
			y = []
			for j, p in enumerate(pairs):
				x.append(to_onehot(p[0], vocab_size))
				y.append(to_onehot(p[1], vocab_size))
				
			x = np.array(x)
			y = np.array(y)
			loss += model.train_on_batch(x, y)
		print(loss)
		history += str(loss) + "\n"
	return model, history


def load_corpus(net_name):
	corpus_0 = pd.read_csv("../data/polygon_clean.txt", sep="\n")
	corpus_0 = corpus_0['title']

	corpus_1 = pd.read_csv("../data/abcnews_clean.txt", sep="\n")
	corpus_1 = corpus_1['title']

	corpus_2 = pd.read_csv("../data/nytimes_clean.txt", sep="\n")
	corpus_2 = corpus_2['title']

	if net_name == 'poly':
		corpus = corpus_0
	elif net_name == 'abc':
		corpus = corpus_1
	elif net_name == 'ny':
		corpus = corpus_2
	elif net_name == 'nap':
		corpus = pd.concat([corpus_0, corpus_1, corpus_2])
	else:
		print("invalid net name")
		exit()
	return corpus


def tokenize(corpus):
	vocab_size = 5000
	tokenizer = Tokenizer(num_words=vocab_size, filters='"#%&()*+-;:<=>@[\\]`\'{|}~\t\n', split=" ", lower=True)
	tokenizer.fit_on_texts(corpus)
	vocab_size = len(tokenizer.word_index) + 1
	word_index = tokenizer.word_index
	print(word_index)
	print("vocabulary size: ", vocab_size)
	rword_index = {v: k for k, v in word_index.items()}
	return tokenizer, word_index, rword_index, vocab_size


#setup
net_name = sys.argv[1]
epochs = int(sys.argv[2])

#loading and processing data
corpus = load_corpus(net_name)
tokenizer, word_index, rword_index, vocab_size = tokenize(corpus)

#save tokenizer
f_token_name = "token_" + net_name + ".pkl"
dump(tokenizer, open(f_token_name, 'wb'))

#creating and training network
model = create()
model, history = train(model, tokenizer, vocab_size, epochs)

#saving data
f_hist_name = "hist_" + net_name + "_" + str(epochs) + ".dat"
f_hist = open(f_hist_name, "w")
f_hist.write(history)
f_hist.close()
f_model_name = "skip_" + net_name + "_" + str(epochs) + ".h5"
model.save(f_model_name)
