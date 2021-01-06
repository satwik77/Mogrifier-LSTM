import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import unicodedata
from collections import OrderedDict


class Corpus(object):
	def __init__(self, path, voc, debug=False):
		self.voc = voc
		self.debug= debug
		self.data= self.create_ids(path)

	def create_ids(self, path):
		assert os.path.exists(path)

		id_tensors = []
		with open(path,'r', encoding='utf8') as f:
			for i, line in enumerate(f):
				words = line.split()
				ids = [self.voc.get_id(w) for w in words] + [self.voc.get_id('</s>')]
				id_tensors+= ids
				if self.debug:
					if i>500:
						break

		id_tensors = torch.tensor(id_tensors).type(torch.int64)
		return id_tensors



	def process_string(self, string):
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " 's", string)
		string = re.sub(r"\'ve", " 've", string)
		string = re.sub(r"n\'t", " n't", string)
		string = re.sub(r"\'re", " 're", string)
		string = re.sub(r"\'d", " 'd", string)
		string = re.sub(r"\'ll", " 'll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " ( ", string)
		string = re.sub(r"\)", " ) ", string)
		string = re.sub(r"\?", " ? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string



class Sampler(object):
	def __init__(self, corpus, batch_size, bptt=None):
		self.corpus= corpus
		self.batch_size = batch_size
		self.bptt= bptt
		self.data = self.batchify(corpus.data, batch_size)
		self.num_batches = self.data.size(0) // self.bptt

	def batchify(self, data, bsz):
		# Work out how cleanly we can divide the dataset into bsz parts.
		nbatch = data.size(0) // bsz
		# Trim off any extra elements that wouldn't cleanly fit (remainders).
		data = data.narrow(0, 0, nbatch * bsz)
		# Evenly divide the data across the bsz batches.
		data = data.view(bsz, -1).t().contiguous()
		return data

	def get_batch(self, i):
		seq_len = min(self.bptt, len(self.data) - 1 -i)
		source = self.data[i: i+seq_len]
		target = self.data[i+1:i+1+seq_len]

		return source, target

	def __len__(self):
		return len(self.data)


class TextDataset(Dataset):
	'''
		Expecting csv files with columns ['sent', 'label']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train:
						is_bert: if the data sampled should be fed to a BERT based model
						bert_model: Only relevant when is_bert is True, speicifies the type of bert model to which data is to be fed 

	'''

	def __init__(self, data_path='./data/', dataset='ptb', datatype='train', max_length=30, is_debug=False, is_train=False):
		if datatype=='train':
			file_path = os.path.join(data_path, dataset, 'train.csv')
		else:
			file_path = os.path.join(data_path, dataset, 'dev.csv')

		self.lines = []

		with open(file_path, 'r', encoding='utf8') as f:
			for line in f:
				self.lines.append(line)

		if is_debug:
			self.lines= self.lines[:5000:500]

		self.max_length= max_length

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		# sent = self.process_string(self.unicodeToAscii(self.sents[idx]))
		line = self.lines[idx]
		return {'line': self.curb_to_length(line)}


	def curb_to_length(self, string):
		return ' '.join(string.strip().split()[:self.max_length])

	def process_string(self, string):
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " 's", string)
		string = re.sub(r"\'ve", " 've", string)
		string = re.sub(r"n\'t", " n't", string)
		string = re.sub(r"\'re", " 're", string)
		string = re.sub(r"\'d", " 'd", string)
		string = re.sub(r"\'ll", " 'll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " ( ", string)
		string = re.sub(r"\)", " ) ", string)
		string = re.sub(r"\?", " ? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string

	def unicodeToAscii(self, string):
		return ''.join(c for c in unicodedata.normalize('NFD', string)
					   if unicodedata.category(c) != 'Mn')


class BERTTextDataset(TextDataset):

	def __init__(self, data_path='./data/', dataset='cola', datatype='train', max_length=30, is_debug=False, is_train=False, bert_model = 'bert-base-uncased'):
		TextDataset.__init__(self, data_path, dataset, datatype, max_length, is_debug, is_train)
		self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

	def __getitem__(self, idx):
		sent = self.bert_preprocessing(self.unicodeToAscii(self.sents[idx]))
		label = self.labels[idx]

		return {'sent': sent, 'label': label}

	def bert_preprocessing(self, string):
		tokens = self.bert_tokenizer.tokenize(string.strip())
		tokens = tokens[:self.max_length - 2] 	#Pruning to fit the maximum length specification, -2 is kept to accomodate [CLS] and [SEP] in the next step
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		assert len(tokens) <= self.max_length
		return ' '.join(tokens)
