import os
import json
import numpy as np
import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")

import torchaudio
from einops import rearrange, repeat

from submodules.ImageBind.imagebind.models.imagebind_model import imagebind_huge, ModalityType
from submodules.ImageBind.imagebind.data import load_and_transform_text, load_and_transform_vision_data


class CLIPModel(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.net = imagebind_huge(pretrained=True)
	
	def encode_image(self, images):
		assert images.ndim == 4 and list(images.shape[1:]) == [3, 224, 224], images.shape
		embeddings = self.net({
			ModalityType.VISION: images
		})[ModalityType.VISION]
		return embeddings
	
	def encode_audio(self, audios):
		assert audios.ndim == 4 and list(audios.shape[1:]) == [1, 128, 204], audios.shape
		embeddings = self.net({
			ModalityType.AUDIO: audios
		})[ModalityType.AUDIO]
		logit_scale = self.net.modality_postprocessors[ModalityType.AUDIO][1].log_logit_scale
		max_logit_scale = self.net.modality_postprocessors[ModalityType.AUDIO][1].max_logit_scale
		logit_scale = torch.clip(logit_scale.exp(), max=max_logit_scale)
		embeddings = embeddings / logit_scale # logit_scale = log(20)
		return embeddings
	
	def encode_text(self, texts: List[str]):
		transformed_texts = load_and_transform_text(texts, device=torch.device("cuda"))
		
		embeddings = self.net({
			ModalityType.TEXT: transformed_texts
		})[ModalityType.TEXT]
		logit_scale = self.net.modality_postprocessors[ModalityType.TEXT][1].log_logit_scale
		max_logit_scale = self.net.modality_postprocessors[ModalityType.TEXT][1].max_logit_scale
		logit_scale = torch.clip(logit_scale.exp(), max=max_logit_scale)
		embeddings = embeddings / logit_scale  # logit_scale = ?
		return embeddings
	
	def forward(self, images, audios=None, texts=None):
		image_embeddings = self.encode_image(images)
		
		result_dict = {}
		
		if audios is not None:
			audio_embeddings = self.encode_audio(audios)
			ia_sim = (image_embeddings * audio_embeddings).sum(dim=1)
			result_dict["ia_sim"] = ia_sim
		
		if texts is not None:
			text_embeddings = self.encode_text(texts)
			it_sim = (image_embeddings * text_embeddings).sum(dim=1)
			result_dict["it_sim"] = it_sim
		
		return result_dict
	
def load_clip_model():
	clip_model = CLIPModel()
	clip_model.requires_grad_(False)
	clip_model.eval()
	return clip_model


	
	

