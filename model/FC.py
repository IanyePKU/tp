import torch
import torch.nn as nn
import torch.autograd as ag

no_linear_func = nn.Tanh
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
		target = output[self.num_layers] - grad_output[0]
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
			output[i + 1] = out.clone().detach()

		croped_input = {}
		rec_input = {}
		for i in range(self.num_layers - 1):
			croped_input[i + 1] = output[i + 1] + torch.randn(output[i + 1].shape).to(device) * (0.36133) #output[i + 1].mean() / 10)
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
		target = output[self.num_layers].clone().detach() - grad_output[0].clone().detach()
		targets[self.num_layers] = target

		for i in range(self.num_layers - 1):
			target = self.backproject[self.num_layers - 1 - i](target) \
						+ output[self.num_layers - 1 - i].clone().detach() \
						- self.backproject[self.num_layers - 1 - i](output[self.num_layers - i].clone().detach())
			targets[self.num_layers - 1 - i] = target
			
		return output, targets


class MLP_ddtp(MLP):
	def __init__(self, channels):
		super(MLP_ddtp, self).__init__(channels)
		self.backproject = {}
		self.bnetwork = nn.Sequential()
		for i in range(self.num_layers - 1):
			self.backproject[i + 1] = nn.Sequential(nn.Linear(channels[-1], channels[i + 1]), no_linear_func())
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
		target = output[self.num_layers].clone().detach() - grad_output[0].clone().detach()
		targets[self.num_layers] = target

		for i in range(self.num_layers - 1):
			tmp_target = self.backproject[self.num_layers - 1 - i](target) \
						+ output[self.num_layers - 1 - i].clone().detach() \
						- self.backproject[self.num_layers - 1 - i](output[self.num_layers].clone().detach())
			targets[self.num_layers - 1 - i] = tmp_target

		return output, targets

	def autoencoder(self, x):
		output = {}
		out = x
		for i in range(self.num_layers):
			out = self.layers[i + 1](out)
			output[i + 1] = out.clone().detach()

		layerL_status = output[self.num_layers] + torch.randn(output[self.num_layers].shape).to(device) * (0.36133)

		one_map_status = {}
		two_map_status = {}

		one_map_status[self.num_layers] = layerL_status
		two_map_status[self.num_layers] = self.layers[self.num_layers](self.backproject[self.num_layers - 1](layerL_status))

		for i in range(self.num_layers - 2):
			one_map_status[i + 2] = self.backproject[i + 2](layerL_status)
			h = self.backproject[i + 1](layerL_status)
			two_map_status[i + 2] = self.layers[i + 2](h)

		return one_map_status, two_map_status

	def get_params_backward(self, i):
		return self.backproject[i].parameters()
