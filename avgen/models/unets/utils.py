from typing import Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention


class InflatedConv3d(nn.Conv2d):
	def forward(self, x):
		video_length = x.shape[2]

		x = rearrange(x, "b c f h w -> (b f) c h w")
		x = super().forward(x)
		x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

		return x


class FFInflatedConv3d(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
		super().__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			**kwargs,
		)
		self.conv_temp = nn.Linear(3 * out_channels, out_channels)
		nn.init.zeros_(self.conv_temp.weight.data)  # initialized to be ones
		nn.init.zeros_(self.conv_temp.bias.data)

	def forward(self, x):
		video_length = x.shape[2]

		x = rearrange(x, "b c f h w -> (b f) c h w")
		x = super().forward(x)
		
		*_, h, w = x.shape
		x = rearrange(x, "(b f) c h w -> (b h w) f c", f=video_length)

		head_frame_index = [0, ] * video_length
		prev_frame_index = torch.clamp(
			torch.arange(video_length) - 1, min=0.0
		).long()
		curr_frame_index = torch.arange(video_length).long()
		conv_temp_nn_input = torch.cat([
			x[:, head_frame_index],
			x[:, prev_frame_index],
			x[:, curr_frame_index]
		], dim=2).contiguous()
		x = x + self.conv_temp(conv_temp_nn_input)
		
		x = rearrange(x, "(b h w) f c -> b c f h w", h=h, w=w)

		return x


class FFAttention(Attention):
	r"""
	A cross attention layer.

	Parameters:
		query_dim (`int`): The number of channels in the query.
		cross_attention_dim (`int`, *optional*):
			The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
		heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
		dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
		dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
		bias (`bool`, *optional*, defaults to False):
			Set to `True` for the query, key, and value linear layers to contain a bias parameter.
	"""
	
	def __init__(
			self,
			*args,
			scale_qk: bool = True,
			processor: Optional["FFAttnProcessor"] = None,
			**kwargs
	):
		super().__init__(*args, scale_qk=scale_qk, processor=processor, **kwargs)
		# set attention processor
		# We use the AttnProcessor by default when torch 2.x is used which uses
		# torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
		# but only if it has the default `scale` argument.
		if processor is None:
			processor = FFAttnProcessor()
		self.set_processor(processor)
	
	def forward(self, hidden_states, video_length, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
		# The `Attention` class can call different attention processors / attention functions
		# here we simply pass along all tensors to the selected processor class
		# For standard processors that are defined here, `**cross_attention_kwargs` is empty
		return self.processor(
			self,
			hidden_states,
			encoder_hidden_states=encoder_hidden_states,
			attention_mask=attention_mask,
			video_length=video_length,
			**cross_attention_kwargs,
		)


class FFAttnProcessor:
	def __init__(self):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError(
				"FFAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
	
	def __call__(self, attn: Attention, hidden_states, video_length, encoder_hidden_states=None, attention_mask=None):
		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)
		inner_dim = hidden_states.shape[-1]
		
		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
		
		query = attn.to_q(hidden_states)
		
		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
		
		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)
		
		# sparse causal attention
		former_frame_index = torch.arange(video_length) - 1
		former_frame_index[0] = 0
		
		key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
		key = key[:, [0] * video_length].contiguous()
		key = rearrange(key, "b f d c -> (b f) d c")
		
		value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
		value = value[:, [0] * video_length].contiguous()
		value = rearrange(value, "b f d c -> (b f) d c")
		
		head_dim = inner_dim // attn.heads
		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		
		# the output of sdp = (batch, num_heads, seq_len, head_dim)
		hidden_states = F.scaled_dot_product_attention(
			query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
		)
		
		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = hidden_states.to(query.dtype)
		
		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)
		return hidden_states