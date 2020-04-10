import torch
import torch.nn as nn
import torch.autograd as ag

no_linear_func = nn.Tanh

class MLP(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.channels = channels
		self.layers  = {}
		self.fnetwork = nn.Sequential()
		for i in range(self.num_layers):
			self.layers[i + 1] = nn.Sequential(nn.Linear(channels[i], channels[i + 1]), no_linear_func())
			self.fnetwork.add_module(f"flayer{i+1}", self.layers[i+1])

		self.init_modules()

	def forward(self, x):
		output = {}
		for i in range(self.num_layers):
			x = self.layers[i + 1](x)
			output[i + 1] = x
		return x

	def init_modules(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
				m.bias.data.zero_()

	def get_params_forward(self, i):
		assert i in self.layers.keys()
		return self.layers[i].parameters()

	@property
	def num_layers(self):
		return len(self.channels) - 1


class MLP_naivetp(MLP):
	def __init__(self, channels):
		super(MLP_naivetp, self).__init__(channels)
		self.backproject = {}
		self.bnetwork = nn.Sequential()
		for i in range(self.num_layers - 1):
			self.backproject[i + 1] = nn.Sequential(nn.Linear(channels[i + 2], channels[i + 1]), no_linear_func())
			self.bnetwork.add_module(f"blayer{i+1}", self.backproject[i + 1])

		self.init_modules()

	def forward_target(self, x, y, loss_func):
		output = {}
		targets = {}
		out = x
		for i in range(self.num_layers):
			out = self.layers[i + 1](out)
			output[i + 1] = out

		loss = loss_func(out, y)
		grad_output = ag.grad(loss, output[self.num_layers])
		assert len(grad_output) == 1
		target = grad_output[0] + output[self.num_layers]
		targets[self.num_layers] = target

		for i in range(self.num_layers - 1):
			target = self.backproject[self.num_layers - 1 - i](target)
			targets[self.num_layers - 1 - i] = target
			
		return output, targets

	def autoencoder(self, x):
		output = {}
		out = x
		for i in range(self.num_layers):
			out = self.layers[i + 1](out)
			output[i + 1] = out

		croped_input = {}
		rec_input = {}
		for i in range(self.num_layers - 1):
			croped_input[i + 1] = output[i + 1] + torch.randn(output[i + 1].shape) * (output[i + 1].mean() / 10)
			h = self.layers[i + 2](croped_input[i + 1])
			rec_input[i + 1] = self.backproject[i + 1](h)

		return croped_input, rec_input

	def get_params_backward(self, i):
		return self.backproject[i].parameters()


class MLP_dtp(MLP_naivetp):
	def __init__(self, channels):
		super(MLP_dtp, self).__init__(channels)

	def forward_target(self, x, y, loss_func):
		output = {}
		targets = {}
		out = x
		for i in range(self.num_layers):
			out = self.layers[i + 1](out)
			output[i + 1] = out

		loss = loss_func(out, y)
		grad_output = ag.grad(loss, output[self.num_layers])
		assert len(grad_output) == 1
		target = grad_output[0] + output[self.num_layers]
		targets[self.num_layers] = target

		for i in range(self.num_layers - 1):
			target = self.backproject[self.num_layers - 1 - i](target) \
						+ (output[self.num_layers - 1 - i] - self.backproject[self.num_layers - 1 - i](output[self.num_layers - i]))
			targets[self.num_layers - 1 - i] = target
			
		return output, targets


class MLP_ddtp(MLP):
	def __init__(self, channels):
		super(MLP_ddtp, self).__init__(channels)
		self.backproject = {}
		for i in range(self.num_layers - 1):
			self.backproject[i + 1] = nn.Linear(channels[-1], channels[i + 1])

	def forward_target(self, x, y, loss_func):
		output = {}
		targets = {}
		for i in range(self.num_layers):
			x = self.layers[i + 1](x)
			output[i + 1] = x

		grad_output = ag.grad
		target = grad_output + output[self.num_layers]
		targets[self.num_layers] = target

		for i in range(self.num_layers - 1):
			targets[self.num_layers - 1 - i] = self.backproject[self.num_layers - 1 - i](target)

		return output, targets

	def autoencoder(self, y, i):
		assert i > 0 and i < self.num_layers
		h_upper_layer = self.backproject[i](y)
		h_lower_layer = self.backproject[i - 1](y)
		h_upper_layer2 = self.layers[i](h_lower_layer)
		return [h_upper_layer, h_upper_layer2]

	def get_params_backward(self, i):
		return self.backproject[i].parameters
