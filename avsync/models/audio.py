# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class Basic2DBlock(nn.Module):
	def __init__(self, in_planes, out_planes, stride=(1, 1)):
		self.__dict__.update(locals())
		super(Basic2DBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=False)
		self.bn1 = nn.BatchNorm2d(out_planes)
		self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False)
		self.bn2 = nn.BatchNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		return x


class AudioConv2DNet(ModelMixin, ConfigMixin):
	
	@register_to_config
	def __init__(
			self,
			pretrained=False
	):
		super(AudioConv2DNet, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)

		self.block1 = Basic2DBlock(64, 64, stride=(2, 2))
		self.block2 = Basic2DBlock(64, 128, stride=(2, 2))
		self.block3 = Basic2DBlock(128, 256, stride=(2, 2))
		self.block4 = Basic2DBlock(256, 512)
		self.out_dim = 512
		
		if pretrained: self.load_pretrained_model()
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		# (b, c, n, t)
		x = x.mean(dim=(2,3))
		return x
	
	def load_pretrained_model(self, ):
		path = "./pretrained/AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar"
		pretrained_state_dict = torch.load(path, map_location=torch.device("cpu"))["model"]
		audio_weights = {}
		for key, val in pretrained_state_dict.items():
			if key.startswith("module.audio_model"):
				audio_weights[key.replace("module.audio_model.", "")] = val
		
		self.load_state_dict(audio_weights)