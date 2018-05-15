import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
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
	model.add(LSTM(128, input_shape=(maxlen, v)))
	model.add(LSTM(128))
	model.add(Dense(v, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimzer=RMSprop(lr=0.01))
	return model

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


def load_dict(net_name):
	#TODO get word_index
	f_name = "dict_" + net_name + ".json"
	json_0 = open(f_name, "r").read()
	word_index = json.loads(json_0)
	#TODO get rword_index
	f_name = "rdict_" + net_name + ".json"
	json_1 = open(f_name, "r").read()
	rword_index = json.loads(json_1)
	vocab_size = len(word_index)
	return word_index, rword_index, vocab_size


def gen_seq(corpus, tokenizer, vocab_size, max_len=3):
	x = []
	y = []

	for i, seq in enumerate(tokenizer.texts_to_sequences(corpus)):
		for j in range(0, len(seq) - max_len):	
			x.append(seq[j: j + max_len])
			y.append(seq[j + max_len])
	
	x_new = []
	for e in x:
		x_new.append(np.array([to_onehot(e_2, vocab_size) for e_2 in e]))
	x = np.array(x_new)
	y = np.array([to_onehot(e, vocab_size) for e in y])
	return x, y


def train(model, x, y, epochs, net_name):
	for i in range(0, 5):
		hist = model.fit(x, y, batch_size=128, epochs=epochs/5)
		print(hist)
		curr_epoch = i * epoch / 5
		model_name = "lstm_" + net_name + "_" + int(curr_epoch) + ".h5"
		model.save(model_name)
	return model
	

net_name = sys.argv[1]
epochs = int(sys.argv[2])

corpus = load_corpus(net_name)
tokenizer, word_index, rword_index, vocab_size = tokenize(corpus)
x, y = gen_seq(corpus, tokenizer, vocab_size)

model = create(vocab_size, 3)
model = train(model)



















