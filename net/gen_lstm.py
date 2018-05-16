import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from pickle import dump
import pandas as pd
import numpy as np
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
	model.add(LSTM(128, input_shape=(max_len, v), return_sequences=True))
	model.add(LSTM(128))
	model.add(Dense(v, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
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




def gen_seq(corpus, tokenizer, vocab_size, max_len=3):
	x = []
	y = []

	for i, seq in enumerate(tokenizer.texts_to_sequences(corpus)):
		for j in range(0, len(seq) - max_len):	
			x.append(seq[j: j + max_len])
			y.append(seq[j + max_len])
	
	x_new = []
	for e in x:
		x_new.append(np.array([to_onehot(e_2, vocab_size) for e_2 in e], dtype=np.bool))
	x = np.array(x_new, dtype=np.bool)
	y = np.array([to_onehot(e, vocab_size) for e in y], dtype=np.bool)
	return x, y


def train(model, x, y, epochs, net_name):
	history = ""
	for i in range(0, epochs):
		loss = model.train_on_batch(x, y)
		print(loss)
		if i % 10 == 0:
			history += str(i) + ":" + str(loss) + "\n"
			model_name = "lstm_" + net_name + "_" + str(epochs) + ".h5"
			model.save(model_name)
	return model, history
	

net_name = sys.argv[1]
epochs = int(sys.argv[2])

corpus = load_corpus(net_name)
tokenizer, word_index, rword_index, vocab_size = tokenize(corpus)


f_token_name = "token_" + net_name + ".pkl"
#tokenizer = load(open(f_token_name, 'rb'))


x, y = gen_seq(corpus, tokenizer, vocab_size)
print(len(y))
exit()

model = create(vocab_size, 3)
model, hist = train(model, x, y, epochs, net_name)

f_hist_name = "lstm_" + net_name + ".dat"
f = open(f_hist_name, "w")
f.write(hist)
f.close()



















