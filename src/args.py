import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')


	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
	# parser.add_argument('-debug', action='store_true', help='Operate on debug mode')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Run name should just be alphabetical word (no special characters to be included)
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-display_freq', type=int, default=35, help='number of batches after which to display loss')
	parser.add_argument('-dataset', type=str, default='ptb', help='Dataset')


	# Input files
	parser.add_argument('-vocab_size', type=int, default=50000, help='Vocabulary size to consider')
	# parser.add_argument('-res_file', type=str, default='generations.txt', help='File name to save results in')
	# parser.add_argument('-res_folder', type=str, default='Generations', help='Folder name to save results in')
	# parser.add_argument('-out_dir', type=str, default='out', help='Out Dir')
	# parser.add_argument('-len_sort', action="store_true", help='Sort based on length')
	parser.add_argument('-histogram', dest='histogram', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-histogram', dest='histogram', action='store_false', help='Operate in normal mode')
	parser.set_defaults(histogram=True)


	# Device Configuration
	parser.add_argument('-gpu', type=int, default=1, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=1111, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	# parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

	# Dont modify ckpt_file
	# If you really want to then assign it a name like abc_0.pth.tar (You may only modify the abc part and don't fill in any special symbol. Only alphabets allowed
	# parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')


	# LSTM parameters
	parser.add_argument('-emb_size', type=int, default=500, help='Embedding dimensions of inputs')
	parser.add_argument('-model_type', type=str, default='SARNN', choices= ['RNN', 'SAN', 'Mogrify', 'SARNN'],  help='Model Type: RNN or Transformer or Mogrifier or SARNN')
	parser.add_argument('-cell_type', type=str, default='LSTM', choices= ['LSTM', 'GRU', 'RNN'],  help='RNN cell type, default: lstm')
	# parser.add_argument('-use_attn', action='store_true', help='To use attention mechanism?')
	# parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	parser.add_argument('-hidden_size', type=int, default=500, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=1, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.5, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	# parser.add_argument('-emb_size', type=int, default=256, help='Embedding dimensions of encoder and decoder inputs')
	# parser.add_argument('-beam_width', type=int, default=10, help='Specify the beam width for decoder')
	parser.add_argument('-max_length', type=int, default=35, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-bptt', type=int, default=35, help='Specify bptt length')


	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	parser.add_argument('-tied', dest='tied', action='store_true', help='Tied Weights in input and output embeddings')
	parser.add_argument('-no-tied', dest='tied', action='store_false', help='Tied Weights in input and output embeddings')
	parser.set_defaults(tied=True)

	# parser.add_argument('-bidirectional', dest='bidirectional', action='store_true', help='Bidirectionality in LSTMs')
	# parser.add_argument('-no-bidirectional', dest='bidirectional', action='store_false', help='Bidirectionality in LSTMs')
	# parser.set_defaults(bidirectional=True)



	''' Transformer '''
	parser.add_argument('-d_model', type=int, default=512, help='Embedding size in Transformer')
	parser.add_argument('-d_ffn', type=int, default=1024, help='Hidden size of FFN in Transformer')
	parser.add_argument('-heads', type=int, default=4, help='Number of Attention heads in each layer')
	# parser.add_argument('-use_word2vec', action='store_true', help='Initialization Embedding matrix with word2vec vectors')
	# parser.add_argument('-word2vec_bin', type=str, default='data/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
	# parser.add_argument('-train_word2vec', action='store_true', help='Binary file of word2vec')

	# # BERT parameters
	# parser.add_argument('-bert', action = 'store_true', help = "Whether to use BERT for encoding text")
	# parser.add_argument('-bert_model', type = str, default='bert-base-uncased', help = "Which bert model to use like case or uncased, base or large")
	# parser.add_argument('-freeze_bert', action = 'store_true', help = "Whether to freeze layers of bert or finetune the entire thing")
	# parser.add_argument('-bert_size', type = int, default = 768, help = "Size of hidden representations of BERT")

	# Transformer Model parameters
	# parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	# parser.add_argument('-heads', type=int, default=8, help='Number of Attention Heads')
	# parser.add_argument('-layers', type=int, default=6, help='Number of layers in each encoder and decoder')
	# parser.add_argument('-d_model', type=int, default=512, help='Embedding dimensions of inputs and hidden representations (refer Vaswani et. al)')
	# parser.add_argument('-d_ff', type=int, default=2048, help='Embedding dimensions of intermediate FFN Layer (refer Vaswani et. al)')
	# # parser.add_argument('-beam_width', type=int, default=10, help='Specify the beam width for decoder')
	# parser.add_argument('-max_length', type=int, default=60, help='Specify max decode steps: Max length string to output')
	# parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	# parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	# parser.add_argument('-bidirectional', action='store_true', help='Initialization range for seq2seq model')

	# Training parameters
	parser.add_argument('-lr', type=float, default=20, help='Learning rate')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=64, help='Batch size')
	parser.add_argument('-epochs', type=int, default=100, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='sgd', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
	# # parser.add_argument('-tfr', type=float, default=0.9, help='Teacher forcing ratio')
	# parser.add_argument('-use_word2vec', action='store_true', help='Initialization Embedding matrix with word2vec vectors')
	# # parser.add_argument('-word2vec_bin', type=str, default='/datadrive/satwik/global_data/glove.840B.300d.txt', help='Binary file of word2vec')
	# parser.add_argument('-word2vec_bin', type=str, default='/datadrive/satwik/global_data/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
	# parser.add_argument('-train_word2vec', dest='train_word2vec', action='store_true', help='train word2vec')
	# parser.add_argument('-no-train_word2vec', dest='train_word2vec', action='store_false', help='Do not train word2vec')
	# parser.set_defaults(train_word2vec=True)


	return parser
