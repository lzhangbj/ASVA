# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional
from einops import rearrange

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention
from diffusers.models.attention import FeedForward, AdaLayerNorm, AdaLayerNormZero
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from ..utils import FFAttention


@dataclass
class SpatioTempTransformer3DModelOutput(BaseOutput):
	sample: torch.Tensor


if is_xformers_available():
	import xformers
	import xformers.ops
else:
	xformers = None


class FFSpatioAudioTempTransformer3DModel(ModelMixin, ConfigMixin):
	
	@register_to_config
	def __init__(
			self,
			num_attention_heads: int = 16,
			attention_head_dim: int = 88,
			in_channels: Optional[int] = None,
			num_layers: int = 1,
			dropout: float = 0.0,
			norm_num_groups: int = 32,
			cross_attention_dim: Optional[int] = None,
			audio_cross_attention_dim: Optional[int] = None,
			attention_bias: bool = False,
			activation_fn: str = "geglu",
			num_embeds_ada_norm: Optional[int] = None,
			use_linear_projection: bool = False,
			only_cross_attention: bool = False,
			upcast_attention: bool = False,
	):
		super().__init__()
		self.use_linear_projection = use_linear_projection
		self.num_attention_heads = num_attention_heads
		self.attention_head_dim = attention_head_dim
		inner_dim = num_attention_heads * attention_head_dim
		
		# Define input layers
		self.in_channels = in_channels
		
		self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
		if use_linear_projection:
			self.proj_in = nn.Linear(in_channels, inner_dim)
		else:
			self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
		
		# Define transformers blocks
		self.transformer_blocks = nn.ModuleList(
			[
				BasicTransformerBlock(
					inner_dim,
					num_attention_heads,
					attention_head_dim,
					dropout=dropout,
					cross_attention_dim=cross_attention_dim,
					audio_cross_attention_dim=audio_cross_attention_dim,
					activation_fn=activation_fn,
					num_embeds_ada_norm=num_embeds_ada_norm,
					attention_bias=attention_bias,
					only_cross_attention=only_cross_attention,
					upcast_attention=upcast_attention,
				)
				for d in range(num_layers)
			]
		)
		
		# 4. Define output layers
		if use_linear_projection:
			self.proj_out = nn.Linear(in_channels, inner_dim)
		else:
			self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
	
	def forward(
			self,
			hidden_states,
			encoder_hidden_states=None,
			audio_encoder_hidden_states=None,
			audio_attention_mask=None,
			timestep=None,
			class_labels=None,
			cross_attention_kwargs=None,
			return_dict: bool = True
	):
		# Input
		assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
		video_length = hidden_states.shape[2]
		hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
		encoder_hidden_states = rearrange(encoder_hidden_states, 'b f n c -> (b f) n c')
		audio_encoder_hidden_states = rearrange(audio_encoder_hidden_states, 'b f n c -> (b f) n c')
		if audio_attention_mask is not None:
			audio_attention_mask = rearrange(audio_attention_mask, 'b f n -> (b f) 1 n')
		
		batch, channel, height, weight = hidden_states.shape
		residual = hidden_states
		
		hidden_states = self.norm(hidden_states)
		if not self.use_linear_projection:
			hidden_states = self.proj_in(hidden_states)
			inner_dim = hidden_states.shape[1]
			hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
		else:
			inner_dim = hidden_states.shape[1]
			hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
			hidden_states = self.proj_in(hidden_states)
		
		# Blocks
		for block in self.transformer_blocks:
			hidden_states = block(
				hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				audio_encoder_hidden_states=audio_encoder_hidden_states,
				audio_attention_mask=audio_attention_mask,
				timestep=timestep,
				video_length=video_length,
				cross_attention_kwargs=cross_attention_kwargs,
				class_labels=class_labels
			)
		
		# Output
		if not self.use_linear_projection:
			hidden_states = (
				hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
			)
			hidden_states = self.proj_out(hidden_states)
		else:
			hidden_states = self.proj_out(hidden_states)
			hidden_states = (
				hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
			)
		
		output = hidden_states + residual
		
		output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
		if not return_dict:
			return (output,)
		
		return SpatioTempTransformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
	def __init__(
			self,
			dim: int,
			num_attention_heads: int,
			attention_head_dim: int,
			dropout=0.0,
			cross_attention_dim: Optional[int] = None,
			audio_cross_attention_dim: Optional[int] = None,
			activation_fn: str = "geglu",
			num_embeds_ada_norm: Optional[int] = None,
			attention_bias: bool = False,
			only_cross_attention: bool = False,
			double_self_attention: bool = False,
			upcast_attention: bool = False,
			norm_elementwise_affine: bool = True,
			norm_type: str = "layer_norm",
			final_dropout: bool = False,
	):
		super().__init__()
		self.only_cross_attention = only_cross_attention
		
		self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
		self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
		
		if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
			raise ValueError(
				f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
				f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
			)
		
		# Define 3 blocks. Each block has its own normalization layer.
		# 1. SC-Cross-Attn
		if self.use_ada_layer_norm:
			self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
		elif self.use_ada_layer_norm_zero:
			self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
		else:
			self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
		self.attn1 = FFAttention(
			query_dim=dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			dropout=dropout,
			bias=attention_bias,
			cross_attention_dim=cross_attention_dim if only_cross_attention else None,
			upcast_attention=upcast_attention,
		)
		
		# 2. Audio Conditioned Cross-Attn
		self.norm_audio = (
			AdaLayerNorm(dim, num_embeds_ada_norm)
			if self.use_ada_layer_norm
			else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
		)
		self.attn_audio = Attention(
			query_dim=dim,
			cross_attention_dim=audio_cross_attention_dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			dropout=dropout,
			bias=attention_bias,
			upcast_attention=upcast_attention,
		)
		
		# 3. Cross-Attn
		if cross_attention_dim is not None or double_self_attention:
			# We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
			# I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
			# the second cross attention block.
			self.norm2 = (
				AdaLayerNorm(dim, num_embeds_ada_norm)
				if self.use_ada_layer_norm
				else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
			)
			self.attn2 = Attention(
				query_dim=dim,
				cross_attention_dim=cross_attention_dim if not double_self_attention else None,
				heads=num_attention_heads,
				dim_head=attention_head_dim,
				dropout=dropout,
				bias=attention_bias,
				upcast_attention=upcast_attention,
			)  # is self-attn if encoder_hidden_states is none
		else:
			self.norm2 = None
			self.attn2 = None
		
		# 4. Temp-Attn
		self.pos_proj_temp = Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)
		self.pos_embedding_temp = TimestepEmbedding(
			dim,
			dim,
			act_fn="silu",
			post_act_fn=None,
			cond_proj_dim=None,
		)
		
		self.attn_temp = Attention(
			query_dim=dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			dropout=dropout,
			bias=attention_bias,
			upcast_attention=upcast_attention,
		)
		nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
		self.norm_temp = (
			AdaLayerNorm(dim, num_embeds_ada_norm)
			if self.use_ada_layer_norm
			else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
		)
		
		# 5. Feed-forward
		self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
		self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
	
	def forward(
			self,
			hidden_states,
			attention_mask=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			audio_encoder_hidden_states=None,
			audio_attention_mask=None,
			timestep=None,
			video_length=None,
			cross_attention_kwargs=None,
			class_labels=None,
	):
		# Notice that normalization is always applied before the real computation in the following blocks.
		# 1. Self-Attention
		if self.use_ada_layer_norm:
			norm_hidden_states = self.norm1(hidden_states, timestep)
		elif self.use_ada_layer_norm_zero:
			norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
				hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
			)
		else:
			norm_hidden_states = self.norm1(hidden_states)
		
		cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
		attn_output = self.attn1(
			norm_hidden_states,
			encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
			attention_mask=attention_mask,
			video_length=video_length,
			**cross_attention_kwargs,
		)
		if self.use_ada_layer_norm_zero:
			attn_output = gate_msa.unsqueeze(1) * attn_output
		hidden_states = attn_output + hidden_states
		
		# 2. Audio Cross-Attention
		if self.attn_audio is not None:
			norm_hidden_states = (
				self.norm_audio(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_audio(hidden_states)
			)
			attn_output = self.attn_audio(
				norm_hidden_states,
				encoder_hidden_states=audio_encoder_hidden_states,
				attention_mask=audio_attention_mask,
				**cross_attention_kwargs,
			)
			hidden_states = attn_output + hidden_states
		
		# 3. Cross-Attention
		if self.attn2 is not None:
			norm_hidden_states = (
				self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
			)
			# TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
			# prepare attention mask here
			
			attn_output = self.attn2(
				norm_hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=encoder_attention_mask,
				**cross_attention_kwargs,
			)
			hidden_states = attn_output + hidden_states
		
		# 3. Temporal-Attention
		
		# Add positional embedding
		device = hidden_states.device
		dtype = hidden_states.dtype
		pos_embed = self.pos_proj_temp(torch.arange(video_length).long()).to(device=device, dtype=dtype)  # (f c)
		pos_embed = self.pos_embedding_temp(pos_embed).unsqueeze(0)  # (1, f, c)
		
		seq_len = hidden_states.shape[1]
		hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
		norm_hidden_states = (
			self.norm_temp(hidden_states + pos_embed, timestep) if self.use_ada_layer_norm else self.norm_temp(
				hidden_states + pos_embed)
		)
		hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
		hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=seq_len)
		
		# 4. Feed-forward
		norm_hidden_states = self.norm3(hidden_states)
		
		if self.use_ada_layer_norm_zero:
			norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
		
		ff_output = self.ff(norm_hidden_states)
		
		if self.use_ada_layer_norm_zero:
			ff_output = gate_mlp.unsqueeze(1) * ff_output
		
		hidden_states = ff_output + hidden_states
		
		return hidden_states

