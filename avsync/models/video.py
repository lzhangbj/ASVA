# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class BasicR2P1DBlock(nn.Module):
	def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
		super(BasicR2P1DBlock, self).__init__()
		spt_stride = (1, stride[1], stride[2])
		tmp_stride = (stride[0], 1, 1)
		self.spt_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=spt_stride, padding=(0, 1, 1), bias=False)
		self.spt_bn1 = nn.BatchNorm3d(out_planes)
		self.tmp_conv1 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=tmp_stride, padding=(1, 0, 0), bias=False)
		self.tmp_bn1 = nn.BatchNorm3d(out_planes)

		self.spt_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
		self.spt_bn2 = nn.BatchNorm3d(out_planes)
		self.tmp_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
		self.out_bn = nn.BatchNorm3d(out_planes)

		self.relu = nn.ReLU(inplace=True)

		if in_planes != out_planes or any([s!=1 for s in stride]):
			self.res = True
			self.res_conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0), bias=False)
		else:
			self.res = False

	def forward(self, x):
		x_main = self.tmp_conv1(self.relu(self.spt_bn1(self.spt_conv1(x))))
		x_main = self.relu(self.tmp_bn1(x_main))
		x_main = self.tmp_conv2(self.relu(self.spt_bn2(self.spt_conv2(x_main))))

		x_res = self.res_conv(x) if self.res else x
		x_out = self.relu(self.out_bn(x_main + x_res))
		return x_out


class VideoR2Plus1DNet(ModelMixin, ConfigMixin):
	"""
	Adapted from https://github.com/facebookresearch/VMZ/blob/4c14ee6f8eae8e2ac97fc4c05713b8a112eb1f28/lib/models/video_model.py
	Adaptation has a full Conv3D stem, and does not adjust for the number of dimensions between the spatial and temporal convolution.
	"""
	@register_to_config
	def __init__(
			self,
			pretrained=False
	):
		super(VideoR2Plus1DNet, self).__init__()
		
		self.conv1 = nn.Sequential(
			nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
		)
		self.conv2x = nn.Sequential(BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64))
		self.conv3x = nn.Sequential(BasicR2P1DBlock(64, 128, stride=(2, 2, 2)), BasicR2P1DBlock(128, 128))
		self.conv4x = nn.Sequential(BasicR2P1DBlock(128, 256, stride=(2, 2, 2)), BasicR2P1DBlock(256, 256))
		self.conv5x = nn.Sequential(BasicR2P1DBlock(256, 512, stride=(2, 2, 2)), BasicR2P1DBlock(512, 512))
		self.out_dim = 512
		
		if pretrained:
			self.load_pretrained_model()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2x(x)
		x = self.conv3x(x)
		x = self.conv4x(x)
		x = self.conv5x(x)
		# (b c f h w)
		x = x.mean(dim=(2,3,4))
		return x

	def load_pretrained_model(self):
		path = "./pretrained/AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar"
		pretrained_state_dict = torch.load(path, map_location=torch.device("cpu"))["model"]
		audio_weights = {}
		for key, val in pretrained_state_dict.items():
			if key.startswith("module.video_model"):
				audio_weights[key.replace("module.video_model.", "")] = val
				
		self.load_state_dict(audio_weights)