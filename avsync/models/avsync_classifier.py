import os

import torch.nn as nn

from avsync.models.audio import AudioConv2DNet
from avsync.models.video import VideoR2Plus1DNet
from avsync.models.head import FCHead


class AVSyncClassifier(nn.Module):
	
	def __init__(
			self,
			audio_encoder,
			video_encoder,
			head
	):
		super().__init__()
		self.audio_encoder = audio_encoder
		self.video_encoder = video_encoder
		self.head = head
	
	def forward(self, audio, video):
		'''
		@audio: (b c n t)
		@video: (b c f h w)
		'''
		score = self.head(
			self.audio_encoder(audio),
			self.video_encoder(video)
		)[:, 0]
		
		return score


def load_avsync_model(model_path="checkpoints/avsync/vggss_sync_contrast/ckpts/checkpoint-40000/modules"):
	audio_encoder = AudioConv2DNet.from_pretrained(
		os.path.join(model_path, "audio_encoder"), use_safetensors=False
	)
	video_encoder = VideoR2Plus1DNet.from_pretrained(
		os.path.join(model_path, "video_encoder"), use_safetensors=False
	)
	head = FCHead.from_pretrained(
		os.path.join(model_path, "head"), use_safetensors=False
	)
	
	avsync_net = AVSyncClassifier(audio_encoder, video_encoder, head)
	avsync_net.eval()
	avsync_net.requires_grad_(False)
	
	return avsync_net
