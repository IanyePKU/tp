import os

import torch
import torchvision.datasets
import torch.nn as nn
import model.FC as FC
from model.get_loss import *
import torchvision.transforms as transforms
import torch.optim
import torch.autograd as ag
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_grad_mul(losses, losses_map, model):

	def set_grad(loss, param_gens):
		params_list = []
		for param_gen in param_gens:
			for param in param_gen:
				params_list.append(param)
		tuple_grad = ag.grad(loss, params_list, retain_graph=True)
		for i, param in enumerate(params_list):
			param.grad = tuple_grad[i]
			# print(param.grad)

	for loss_idx in losses_map:
		if loss_idx not in losses.keys():
			print(f"unknown key: {loss_idx}")
			continue
		loss = losses[loss_idx]
		set_grad(loss, [model.get_params_forward(i) for i in losses_map[loss_idx][0]] \
				+ [model.get_params_backward(i) for i in losses_map[loss_idx][1]])


def set_grad_zero(model):
	for p in model.parameters():
		if p.grad is not None:
			p.grad.zero_()


def train(model, data_loader, cfg):
	print("start training!!!")
	print(model)
	opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

	model.train()
	last_iter = 0
	for epoch in range(cfg.epoch):
		for step, (images, labels) in enumerate(data_loader["train"]):
			images, labels = images.to(device), labels.to(device)
			images = images.reshape(-1, 28 * 28) # ?????
			outputs = {"forward": {}, "autoencoder": {}}
			targets = {"forward": {}, "autoencoder": {}}

			if cfg.tp:
				outputs["forward"], targets["forward"] = model.forward_target(images, labels, nn.CrossEntropyLoss())
				outputs["autoencoder"], targets["autoencoder"] = model.autoencoder(images)
				loss_func = nn.MSELoss()
				losses = get_tp_losses(outputs, targets, loss_func)
			else:
				loss_func = nn.CrossEntropyLoss()
				outputs= model.forward(images)
				losses = get_bp_losses(outputs, labels, loss_func)

			# print(losses)
			# exit()

			if (last_iter + 1) % cfg.log_step == 0:
				print(f"epoch: {epoch}, step: {step}, loss: {get_losses_sum(losses)}")

			set_grad_zero(model)
			set_grad_mul(losses, cfg.losses_map, model)
			opt.step()
		
			last_iter += 1
			# print(step)

		pre = -1
		correct = 0
		for step, (images, labels) in enumerate(data_loader["test"]):
			images, labels = images.to(device), labels.to(device)
			images = images.reshape(-1, 28 * 28) # ?????
			outputs = model.forward(images).flatten()
			correct += int(torch.argmax(outputs) == labels.flatten())

		pre = correct / len(data_loader["test"])
		print(pre)


def get_dataloader(cfg):
	root = '../data'
	name_2_datagen = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10}
	assert cfg.name in name_2_datagen.keys()
	dataset_gen = name_2_datagen[cfg.name]

	#train_dataset = dataset_gen(root, train=True, transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), download=True)
	#test_dataset = dataset_gen(root, train=False, transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), download=True)
	
	train_dataset = dataset_gen(root, train=True, transform=transforms.ToTensor(), download=True)
	test_dataset = dataset_gen(root, train=False, transform=transforms.ToTensor(), download=True)

	train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size,
								  shuffle=True, num_workers=cfg.workers, pin_memory=True)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, 
								  shuffle=False, num_workers=cfg.test_workers, pin_memory=True)

	return {"train": train_dataloader, "test": test_dataloader}


def get_model(cfg):
	assert cfg.name in FC.__dict__.keys()
	model_gen = FC.__dict__[cfg.name]
	model = model_gen(**cfg.kwargs)
	return model.to(device)


def main():
	import argparse
	parser = argparse.ArgumentParser("fsl_name_args")
	parser.add_argument("--config", type=str, default="./experments/mlp_bp.yaml")
	args = parser.parse_args()

	# args.config = os.path.join(os.getcwd(), "experiments", "mlp_naivetp.yaml")

	with open(args.config) as f:
		config = yaml.load(f)
	cfg = EasyDict(config)

	data_loader = get_dataloader(cfg.dataset)
	model = get_model(cfg.model)
	train(model, data_loader, cfg.train)


if __name__ == "__main__":
	main()