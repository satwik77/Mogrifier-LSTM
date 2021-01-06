import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftClassifier(nn.Module):
	'''
	Takes hidden representation from encoder as input 
	and generates classifies into correponding classes

	Args:
			hidden_size: Dimension of output from encoder
			output_size: Number of classes
			transform: 	Whether to apply non-linear transformation to encoded representation

	'''

	def __init__(self, hidden_size, output_size, transform=False):
		super(SoftClassifier, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.transform = transform

		if self.transform:
			self.nonlinear = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
			)

		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, hidden):
		output = hidden
		if self.transform:
			output = self.nonlinear(output)
		output = self.out(output)
		output = F.log_softmax(output, dim=1)

		return output
