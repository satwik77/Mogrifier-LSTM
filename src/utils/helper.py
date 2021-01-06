import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import os
import sys
import re
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU
	'''
	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device


def create_save_directories(log_path, req_path):
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	if not os.path.exists(req_path):
		os.makedirs(req_path)


def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			model state
			epoch number
			logger variable
			directory to save models
			checkpoint name
	'''
	ckpt_path = os.path.join(model_path, '{}_{}.pt'.format(ckpt, epoch))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)


def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path: including the run_name
			logger variable: to log messages
		Returns:
			checkpoint: path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		logger.warning('No Checkpoints Found')

		return None
	else:
		latest_epoch = max([int(x.split('_')[-1].split('.')[0]) for x in ckpts])
		ckpts = sorted(ckpts, key= lambda x: int(x.split('_')[-1].split('.')[0]) , reverse=True )
		ckpt_path = ckpts[0]
		logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path

def load_checkpoint(model, mode, ckpt_path, logger, device):
	start_epoch = None
	train_loss = None
	val_loss = None
	voc = None
	score = -1

	try:
		checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		train_loss  =checkpoint['train_loss']
		val_loss = checkpoint['val_loss']
		voc = checkpoint['voc']
		score = checkpoint['val_acc']

		model.to(device)

		if mode == 'train':
			model.train()
		else:
			model.eval()

		logger.info('Successfully Loaded Checkpoint from {}, with epoch number: {} for {}'.format(ckpt_path, start_epoch, mode))

		return start_epoch, train_loss, val_loss, score, voc
	except:
		logger.warning('Could not Load Checkpoint from {}  \t \"at load_checkpoint() in helper.py \"'.format(ckpt_path))
		return start_epoch, train_loss, val_loss, score, voc



class Voc:
	def __init__(self):
		self.trimmed = False
		self.frequented = False
		# self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		# self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		# self.w2c = {'unk':1}
		# self.nwords = 3

		self.w2id= {'</s>': 0}
		self.id2w = {0:'</s>'}
		self.w2c = {}
		self.nwords = 1

	def add_word(self, word):
		if word not in self.w2id:
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			try:
				self.w2c[word] += 1
			except:
				pdb.set_trace()

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def most_frequent(self, topk):
		if self.frequented == True:
			return
		self.frequented = True

		keep_words = []
		count = 3
		sort_by_value = sorted(
			self.w2c.items(), key=lambda kv: kv[1], reverse=True)
		for word, freq in sort_by_value:
			keep_words += [word]*freq
			count += 1
			if count == topk:
				break

		# self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		# self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		# self.w2c = {'unk':1}
		self.w2id= {'</s>': 1}
		self.id2w = {1:'</s>'}
		self.w2c = {}
		self.nwords = 1

		for word in keep_words:
			self.add_word(word)

	def trim(self, mincount):
		if self.trimmed == True:
			return
		self.trimmed = True

		keep_words = []
		for k, v in self.w2c.items():
			if v >= mincount:
				keep_words += [k]*v

		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3
		for word in keep_words:
			self.addWord(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader = None, path=None, debug=False):
		if train_dataloader:
			for data in train_dataloader:
				for sent in data['sent']:
					self.add_sent(sent)
		elif path:
			with open(path, 'r', encoding= 'utf8') as f:
				for i, line in enumerate(f):
					self.add_sent(line)
					if debug:
						if i>500:
							break


		# self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords


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
		string = re.sub(r"\.", " . ", string)
		string = re.sub(r"\(", " ( ", string)
		string = re.sub(r"\)", " ) ", string)
		string = re.sub(r"\?", " ? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string

	# def save_vocab_dict()

