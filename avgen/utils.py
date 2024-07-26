import torch
import torch.nn as nn
import torch.distributed


def get_model_size(model):
	param_size = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
	buffer_size = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
	
	size_all_mb = (param_size + buffer_size) / 1024 ** 2
	return size_all_mb


def freeze_model(model: nn.Module):
	for param in model.parameters():
		param.requires_grad = False


def freeze_and_make_eval(model: nn.Module):
	for param in model.parameters():
		param.requires_grad = False
	model.eval()


def gather_tensor(tensor):
	output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(output_tensors, tensor)
	return torch.cat(output_tensors, dim=0)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		if n == 0: return
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
