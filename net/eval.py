import keras
from keras.models import Sequential, load_model

from pickle import load
import numpy as np
import itertools
import sys
import os


if os.environ['CUDA_VISIBLE_DEVICES'] == '':
	exit()

def to_onehot(i, v):
	x = np.zeros(v, np.float)
	x[i] = 1
	return x


def from_onehot(x):
	return x.argmax()


def vec2word(x, rw_index):
	i = from_onehot(x)
	return rw_index[i]


def word2vec(w, w_index):
	x = w_index[w]
	v = len(w_index) + 1
	return to_onehot(x, v)

def get_top(x, n):
	l = []
	for i in range(0, n):
		l.append(x.argmax())
		x[x.argmax()] = 0
	return np.array(l)


def gen_perm(x, n):
	return np.array(list(itertools.combinations(x, n)))


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def gen_seed(skip_model, word_index, rword_index, v, w):
	x = word2vec(w, word_index)
	x = np.array([x])
	pred = skip_model.predict(x, verbose=0)[0]
	top = get_top(pred, 10)
	seed = [rword_index[w] for w in top if rword_index[w] != 'eos']
	seed_perm = gen_perm(seed, 2)
	seed_0 = np.array([np.insert(e, 0, [w]) for e in seed_perm])
	seed_1 = np.array([np.insert(e, 1, [w]) for e in seed_perm])
	seed_2 = np.array([np.insert(e, 2, [w]) for e in seed_perm])
	seed_perm = np.concatenate([seed_0, seed_1, seed_2])
	return seed_perm

	
def gen_sentences(lstm_model, word_index, rword_index, v, s):
	sentences = list()
	for i in range(0, len(s)):
		ss = s[i]
		x = np.array([word2vec(e, word_index) for e in ss])
		x = np.array([x])
		for j in range(20):
			preds = lstm_model.predict(x, verbose=0)[0]
			#next_index = sample(preds, 1.0)
			next_index = preds.argmax()
			next_word = rword_index[next_index]
			x = x[:,1:,:]
			k = to_onehot(next_index, v)
			k = np.array([[k]])
			x = np.append(x, k, axis=1)
			ss = np.append(ss, next_word)
			if next_word == 'eos':
				break
		sentences.append(ss)
	return np.array(sentences)

def pretty_print(sentences):
	for line in sentences:
		temp = ""
		for word in line:
			if word != 'eos':
				temp += word + " "
		print(temp)


def pretty_write(sentences, net_name, input_word):
	temp = ""
	for line in sentences:
		for word in line:
			if word != 'eos':
				temp += word + " "
		temp += "\n"
	res_name = net_name + "_" + input_word + ".txt"
	f = open(res_name, "w")
	f.write(temp)
	f.close()


net_name = sys.argv[1]
input_word = sys.argv[2]

#load tokenizer + vocab
f_token_name = "token_" + net_name + ".pkl"
tokenizer = load(open(f_token_name, 'rb'))
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
rword_index = {v: k for k, v in word_index.items()}

#load models
f_skip_name = "skip_" + net_name + "_50.h5"
skip_model = load_model(f_skip_name)
f_lstm_name = "lstm_" + net_name + "_100.h5"
lstm_model = load_model(f_lstm_name)


seeds = gen_seed(skip_model, word_index, rword_index, vocab_size, input_word)
sentences = gen_seed(skip_model, word_index, rword_index, vocab_size, input_word)
sentences = gen_sentences(lstm_model, word_index, rword_index, vocab_size, sentences)
pretty_print(sentences)
pretty_write(sentences, net_name, input_word)
