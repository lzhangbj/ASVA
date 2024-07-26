import math
from dataclasses import dataclass
from typing import Optional, TypeVar, Tuple, Any
T = TypeVar('T', bound='Module')
from einops import rearrange, repeat

import numpy as np
import torch
import torch.nn as nn

from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from submodules.ImageBind.imagebind.models import imagebind_model
from submodules.ImageBind.imagebind.models.imagebind_model import ModalityType


@dataclass
class ImageBindSegmaskAudioEncoderOutput(ModelOutput):
	"""
	Args:
		text_embeds(`torch.Tensor` of shape `(batch_size, output_dim`):
			The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
		image_embeds(`torch.Tensor` of shape `(batch_size, output_dim`):
			The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
		text_model_output(`BaseModelOutputWithPooling`):
			The output of the [`CLIPTextModel`].
		vision_model_output(`BaseModelOutputWithPooling`):
			The output of the [`CLIPVisionModel`].
	"""
	audio_embeds: torch.Tensor = None
	audio_encodings: torch.Tensor = None
	audio_segment_masks: torch.BoolTensor = None
	
	def to_tuple(self) -> Tuple[Any]:
		return tuple(self[k] for k in self.keys())


class ImageBindSegmaskAudioEncoder(ModelMixin, ConfigMixin):
	
	@register_to_config
	def __init__(self,
	             n_segment=4,
	             pretrained_model_name="imagebind-huge"
         ):
		super().__init__()
		self.n_segment = n_segment
		
		self.pretrained_model_name = pretrained_model_name
		if pretrained_model_name == "imagebind-huge":
			pretrained_model = imagebind_model.imagebind_huge(pretrained=True)
		
		self.preprocessor = pretrained_model.modality_preprocessors[ModalityType.AUDIO]
		self.trunk = pretrained_model.modality_trunks[ModalityType.AUDIO]
		self.head = pretrained_model.modality_heads[ModalityType.AUDIO]
		self.postprocessor = pretrained_model.modality_postprocessors[ModalityType.AUDIO]
		self.final_layer_norm = nn.LayerNorm(normalized_shape=768, eps=1e-6)
	
	def _auto_split(self, n, n_chunk):
		'''
			automatically split into chunks with n_ele no differ by 1
			if n is not dividible by n_chunk, extra one's will be added to the heading chunks
		'''
		chunk_size = int(math.ceil(n/n_chunk))
		assert chunk_size >= 1, chunk_size
		
		chunk_start_indices = np.round(np.linspace(0, n-chunk_size, n_chunk, endpoint=True)).astype(np.int32)
		
		mask = torch.zeros(n_chunk, n).bool()
		for chunk_index, chunk_start_index in enumerate(chunk_start_indices):
			mask[chunk_index, chunk_start_index:chunk_start_index+chunk_size] = 1
		mask = mask.contiguous()
		assert mask.long().sum() == chunk_size * n_chunk, mask.long().sum()
		
		return mask
	
	def forward(self,
	            input_features: Optional[torch.Tensor],
	            normalize: bool = False,
	            return_dict: Optional[bool] = None):
		
		n_segment = self.n_segment
		
		# 1. reshape to imagebind input
		batchsize = input_features.size(0)
		
		# 2. patchify images and add positional embedding and
		audio_inputs = self.preprocessor(input_features)
		trunk_inputs = audio_inputs["trunk"] # dict of {"tokens": (b, l, d)}
		
		# 3. get audio encoder output
		audio_encodings = self.trunk(**trunk_inputs)  # w/o layer norm (b, seq_len, c)
		head_inputs = audio_inputs["head"]
		cls_embeds = self.head(audio_encodings, **head_inputs)
		# normalize and logit scaling
		if normalize:
			cls_embeds = self.postprocessor(cls_embeds)  # (b, c)
		audio_encodings = self.final_layer_norm(audio_encodings)
		
		# 4. get segment masks
		n, t = 12, 19 # hard code
		segment_mask = self._auto_split(t, n_segment).unsqueeze(1).expand(n_segment, n, t).contiguous() # (s, n, t)
		segment_mask = rearrange(
			segment_mask, "s n t -> s (n t)"
		)
		segment_mask = torch.cat([
			torch.ones(n_segment, 1).bool(),
			segment_mask
		], dim=1) # (s, 1+n*t)
		
		segment_masks = repeat(segment_mask, "n s -> b n s", b=batchsize).contiguous().bool().to(self.device)
		
		if not return_dict:
			return cls_embeds, audio_encodings, segment_masks
		
		return ImageBindSegmaskAudioEncoderOutput(
			audio_embeds=cls_embeds,
			audio_encodings=audio_encodings,
			audio_segment_masks=segment_masks
		)
