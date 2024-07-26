import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class FCHead(ModelMixin, ConfigMixin):
	
	@register_to_config
	def __init__(self, dim=512, out_dim=1, dropout=0.0):
		super().__init__()
		
		self.fc = nn.Sequential(
			nn.Linear(dim*2, dim),
			nn.Dropout(dropout),
			nn.ReLU(inplace=True),
			nn.Linear(dim, dim//2),
			nn.Dropout(dropout),
			nn.ReLU(inplace=True),
			nn.Linear(dim//2, out_dim)
		)
	
	def forward(self, audio_embeddings, video_embeddings):
		
		out = self.fc(torch.cat([audio_embeddings, video_embeddings], dim=1))
		
		return out
