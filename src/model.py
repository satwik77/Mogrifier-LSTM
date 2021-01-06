import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from gensim import models

from src.components.rnns import RNNModel
from src.components.transformers import TransformerModel
from src.components.mogrifierLSTM import MogrifierLSTMModel
from src.components.sa_rnn import SARNNModel

from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint

from collections import OrderedDict



class LanguageModel(nn.Module):
	def __init__(self, config, voc, device, logger):
		super(LanguageModel, self).__init__()

		self.config = config
		self.device = device
		self.logger= logger
		self.voc = voc
		self.lr =config.lr

		self.logger.debug('Initalizing Model...')
		self._initialize_model()

		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 
		# self.criterion = nn.NLLLoss() 
		self.criterion = nn.CrossEntropyLoss(reduction= 'mean') 


	def _initialize_model(self):
		if self.config.model_type == 'RNN':
			self.model = RNNModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)
		elif self.config.model_type == 'SAN':
			self.model = TransformerModel(self.voc.nwords, self.config.d_model, self.config.heads, self.config.d_ffn, self.config.depth, self.config.dropout).to(self.device)
		elif self.config.model_type == 'Mogrify':
			self.model = MogrifierLSTMModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)
		elif self.config.model_type == 'SARNN':
			self.model = SARNNModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)


	def _initialize_optimizer(self):
		self.params = self.model.parameters()

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.25, patience=1, verbose=True)


	def trainer(self, source, targets, hidden, config, device=None ,logger=None):

		self.optimizer.zero_grad()

		if config.model_type == 'RNN':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SAN':
			output = self.model(source)
		elif config.model_type == 'Mogrify':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SARNN':
			output, hidden = self.model(source, hidden)


		loss = self.criterion(output.view(-1,self.voc.nwords), targets.view(-1))
		loss.backward()

		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

		self.optimizer.step()
		# for p in self.model.parameters():
			# p.data.add_(-self.lr, p.grad.data)

		if config.model_type != 'SAN':
			hidden = self.repackage_hidden(hidden)

		return loss.item(), hidden

	def evaluator(self, source, targets, hidden, config, device=None ,logger=None):

		if config.model_type == 'RNN':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SAN':
			output = self.model(source)
		elif config.model_type == 'Mogrify':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SARNN':
			output, hidden = self.model(source, hidden)

		flat_output = output.view(-1, self.voc.nwords)

		loss = self.criterion(flat_output, targets.view(-1))

		if config.model_type != 'SAN':
			hidden = self.repackage_hidden(hidden)

		return  loss.item(), hidden


	def repackage_hidden(self, h):
		"""Wraps hidden states in new Tensors, to detach them from their history."""

		if isinstance(h, torch.Tensor):
			return h.detach()
		else:
			return tuple(self.repackage_hidden(v) for v in h)


def build_model(config, voc, device, logger):
	'''
		Add Docstring
	'''
	model = LanguageModel(config, voc, device, logger)
	model = model.to(device)

	return model



def train_model(model, train_loader, val_loader, voc, device, config, logger, epoch_offset= 0, min_val_loss=float('inf'), min_val_ppl=float('inf'), writer= None):
	'''
		Add Docstring
	'''

	if config.histogram and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)
	
	estop_count=0

	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		val_loss_epoch = 0.0

		# Train Mode
		model.train()

		start_time= time()
		# Batch-wise Training
		if config.model_type != 'SAN':
			hidden = model.model.init_hidden(config.batch_size)
		else:
			hidden = None

		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']
		for batch, i in enumerate(range(0, len(train_loader)-1, config.bptt)):
			source, targets = train_loader.get_batch(i)
			source = source.to(device)
			targets = targets.to(device)

			loss, hidden = model.trainer(source, targets, hidden, config)
			# if batch % config.display_freq==0:
			# 	od = OrderedDict()
			# 	od['Batch'] = batch_num
			# 	od['Loss'] = loss
			# 	print_log(logger, od)

			train_loss_epoch += loss* len(source)

		train_loss_epoch = train_loss_epoch / (len(train_loader)-1)

		time_taken = (time() - start_time)/60.0

		if writer:
			writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

		logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		logger.debug('Starting Validation')
		# pdb.set_trace()

		val_loss_epoch, val_ppl_epoch = run_validation(config, model, val_loader, voc, device, logger)

		if config.opt == 'sgd':
			model.scheduler.step(val_ppl_epoch)

		if val_ppl_epoch < min_val_ppl:
			min_val_loss = val_loss_epoch
			min_val_ppl = val_ppl_epoch

			state = {
				'epoch' : epoch + epoch_offset,
				'model_state_dict': model.state_dict(),
				'voc': model.voc,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss' : train_loss_epoch,
				'val_loss' : min_val_loss,
				'val_ppl': min_val_ppl,
				'lr' : lr_epoch
			}
			logger.debug('Validation Perplexity: {}'.format(val_ppl_epoch))

			save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)
			estop_count=0
		else:
			estop_count+=1

		if writer:
			writer.add_scalar('loss/val_loss', val_loss_epoch, epoch + epoch_offset)
			writer.add_scalar('ppl/val_ppl', val_ppl_epoch, epoch + epoch_offset)

		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['train_loss'] = train_loss_epoch
		od['val_loss']= val_loss_epoch
		od['val_ppl'] = min_val_ppl
		od['lr_epoch'] = lr_epoch
		print_log(logger, od)

		if config.histogram and writer:
			# pdb.set_trace()
			for name, param in model.named_parameters():
				writer.add_histogram(name, param, epoch + epoch_offset)

		# if estop_count > 10:
		# 	logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
		# 	break

	writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
	writer.close()

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		store_results(config, min_val_ppl, min_val_loss, train_loss_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))



def run_validation(config, model, val_loader, voc, device, logger):
	batch_num =1
	val_loss_epoch =0.0
	model.eval()

	if config.model_type != 'SAN':
		hidden = model.model.init_hidden(config.batch_size)
	else:
		hidden = None


	with torch.no_grad():
		for batch, i in enumerate(range(0, len(val_loader)-1, config.bptt)):
				source, targets = val_loader.get_batch(i)
				source =source.to(device)
				targets =targets.to(device)
				loss, hidden = model.evaluator(source, targets, hidden, config)
				val_loss_epoch+= loss*len(source)


	val_loss_epoch = val_loss_epoch/ (len(val_loader)-1)
	val_ppl_epoch = np.exp(val_loss_epoch)

	return val_loss_epoch, val_ppl_epoch




