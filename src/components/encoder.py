import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERTEncoder(nn.Module):

	def __init__(self, model = 'bert-base-uncased', freeze_bert = False):
		super(BERTEncoder, self).__init__()
		self.bert_layer = BertModel.from_pretrained(model)
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False

	def forward(self, input_seqs, input_lengths):

		_, B = input_seqs.shape

		#Prepare attention masks (Find a better way of doing this!)
		attn_masks = input_seqs.transpose(1,0).clone()
		for i in range(B):
			attn_masks[i, :input_lengths[i]] = 1
			attn_masks[i, input_lengths[i]:] = 0
			
		outputs, _ = self.bert_layer(input_seqs.transpose(1,0), attention_mask = attn_masks)
		cls_rep = outputs[:,0] #embedding of [CLS] token

		return outputs.transpose(1,0), cls_rep




class Encoder(nn.Module):
	'''
	Encoder helps in building the sentence encoding module for a batched version
	of data that is sent in [T x B] having corresponding input lengths in [1 x B]

	Args:
			hidden_size: Hidden size of the RNN cell
			embedding: Embeddings matrix [vocab_size, embedding_dim]
			cell_type: Type of RNN cell to be used : LSTM, GRU
			nlayers: Number of layers of LSTM (default = 1)
			dropout: Dropout Rate (default = 0.1)
			bidirectional: Bidirectional model to be formed (default: False)
	'''

	def __init__(self, embedding, hidden_size=512, cell_type='lstm', nlayers=1, dropout=0.1, bidirectional=True):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.nlayers = nlayers
		self.dropout = dropout
		self.embedding = embedding
		self.cell_type = cell_type
		self.embedding_size = self.embedding.embedding_dim
		self.bidirectional = bidirectional

		if self.cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
							   num_layers=self.nlayers,
							   dropout=(0 if self.nlayers == 1 else dropout),
							   bidirectional=bidirectional)
		elif self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)
		else:
			self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  nonlinearity='tanh',							# ['relu', 'tanh']
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)

	def forward(self, input_seqs, input_lengths, hidden=None):
		embedded = self.embedding(input_seqs)
		packed = torch.nn.utils.rnn.pack_padded_sequence(
			embedded, input_lengths)
		outputs, hidden = self.rnn(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
			outputs)  # unpack (back to padded)

		# If bidirectional==True, output_dim = hidden_size*2
		return outputs, hidden
