import os
import sys
import math
import logging
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results
from src.dataloader import Corpus, Sampler
from src.model import LanguageModel, build_model, train_model, run_validation



global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'


def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Create Vocab'''
		train_path = os.path.join(data_path, config.dataset, 'train.txt')
		val_path = os.path.join(data_path, config.dataset, 'val.txt')
		test_path = os.path.join(data_path, config.dataset, 'test.txt')
		voc= Voc()
		voc.create_vocab_dict(config, path= train_path, debug = config.debug)
		voc.create_vocab_dict(config, path= val_path, debug = config.debug)
		voc.create_vocab_dict(config, path= test_path, debug = config.debug)


		'''Load Datasets'''

		train_corpus = Corpus(train_path, voc, debug = config.debug)
		val_corpus = Corpus(val_path, voc, debug = config.debug)
		
		train_loader = Sampler(train_corpus, config.batch_size, config.bptt)
		val_loader = Sampler(val_corpus, config.batch_size, config.bptt)

		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(len(train_corpus.data), len(val_corpus.data))
		logger.info(msg)




		return voc, train_loader, val_loader

	elif config.mode == 'test':
		logger.debug('Loading Test Data...')

		test_set = TextDataset(data_path=data_path, dataset=config.dataset,
							   datatype='dev', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))


def main():
	'''read arguments'''
	parser = build_parser()
	args = parser.parse_args()
	config =args
	mode = config.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	'''Run Config files/paths'''
	run_name = config.run_name
	config.log_path = os.path.join(log_folder, run_name)
	config.model_path = os.path.join(model_folder, run_name)
	config.board_path = os.path.join(board_path, run_name)

	vocab_path = os.path.join(config.model_path, 'vocab.p')
	config_file = os.path.join(config.model_path, 'config.p')
	log_file = os.path.join(config.log_path, 'log.txt')

	if config.results:
		config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

	if is_train:
		create_save_directories(config.log_path, config.model_path)
	else:
		create_save_directories(config.log_path, config.result_path)

	logger = get_logger(run_name, log_file, logging.DEBUG)
	writer = SummaryWriter(config.board_path)

	logger.debug('Created Relevant Directories')
	logger.info('Experiment Name: {}'.format(config.run_name))

	'''Read Files and create/load Vocab'''
	if is_train:

		logger.debug('Creating Vocab and loading Data ...')
		voc, train_loader, val_loader = load_data(config, logger)

		logger.info(
			'Vocab Created with number of words : {}'.format(voc.nwords))		

		with open(vocab_path, 'wb') as f:
			pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)
		logger.info('Vocab saved at {}'.format(vocab_path))



	else:
		test_dataloader = load_data(config, logger)
		logger.info('Loading Vocab File...')

		with open(vocab_path, 'rb') as f:
			voc = pickle.load(f)

		logger.info('Vocab Files loaded from {}'.format(vocab_path))

	# print('Done')

	# TO DO : Load Existing Checkpoints here
	checkpoint = get_latest_checkpoint(config.model_path, logger)


	if is_train:
		
		min_val_loss = torch.tensor(float('inf')).item()
		min_val_ppl = float('inf')
		epoch_offset= 0


		if checkpoint:
			ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
			config.lr = checkpoint['lr']
			model = build_model(config=config, voc=voc, device=device, logger=logger)
			model.load_state_dict(ckpt['model_state_dict'])
			model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
		else:
			model = build_model(config=config, voc=voc, device=device, logger=logger)
		# pdb.set_trace()

		logger.info('Initialized Model')

		with open(config_file, 'wb') as f:
			pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

		logger.debug('Config File Saved')

		logger.info('Starting Training Procedure')
		train_model(model, train_loader, val_loader, voc,
					device, config, logger, epoch_offset, min_val_loss, min_val_ppl, writer)

	else:

		gpu = config.gpu

		with open(config_file, 'rb') as f:
			config = AttrDict(pickle.load(f))
			config.gpu = gpu
			# To do: remove it later
			config.num_labels =2  

		model = build_model(config=config, voc=voc, num_labels=config.num_labels, device=device, logger=logger)

		ep_offset, train_loss, min_val_loss, min_val_ppl, voc = load_checkpoint(
			model, config.mode, checkpoint, logger, device)

		logger.info('Prediction from')
		od = OrderedDict()
		od['epoch'] = ep_offset
		od['train_loss'] = train_loss
		od['val_loss'] = min_val_loss
		od['val_acc'] = min_val_ppl
		print_log(logger, od)

		test_acc_epoch, test_loss_epoch = run_validation(config, model, test_dataloader, voc, device, logger)
		logger.info('Accuracy: {} \t Loss: {}'.format(test_acc_epoch, test_loss_epoch))



if __name__ == '__main__':
	main()




''' Just docstring format '''
# class Vehicles(object):
# 	'''
# 	The Vehicle object contains a lot of vehicles

# 	Args:
# 		arg (str): The arg is used for...
# 		*args: The variable arguments are used for...
# 		**kwargs: The keyword arguments are used for...

# 	Attributes:
# 		arg (str): This is where we store arg,
# 	'''
# 	def __init__(self, arg, *args, **kwargs):
# 		self.arg = arg

# 	def cars(self, distance,destination):
# 		'''We can't travel distance in vehicles without fuels, so here is the fuels

# 		Args:
# 			distance (int): The amount of distance traveled
# 			destination (bool): Should the fuels refilled to cover the distance?

# 		Raises:
# 			RuntimeError: Out of fuel

# 		Returns:
# 			cars: A car mileage
# 		'''
# 		pass