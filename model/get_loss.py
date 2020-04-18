import torch
import torch.nn

def get_bp_losses(outputs, targets, loss_func):
	loss_g = {}
	loss_g["bp_loss"] = loss_func(outputs, targets)
	return loss_g

def get_tp_losses(outputs, targets, loss_func):
	# assert len(outputs) == 2

	loss_g = {}
	output_part1 = outputs["autoencoder"]
	target_part1 = targets["autoencoder"]
	for group in target_part1:
		loss_g[f"autoencoder_layer{group}"] = loss_func[0](output_part1[group], target_part1[group])
	output_part2 = outputs["forward"]
	target_part2 = targets["forward"]
	for group in target_part2:
		loss_g[f"forward_layer{group}"] = loss_func[0](output_part2[group], target_part2[group])
	loss_g["target_loss"] = loss_func[1](outputs["finallayer"], targets["finallayer"])

	return loss_g

def get_losses_sum(losses):
	loss_sum = 0.0
	for loss_name in losses:
		loss_sum += float(losses[loss_name])

	return loss_sum

def get_mulbp_losses(outputs, targets):
	pass