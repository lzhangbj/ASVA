import torch
from torch import nn

from .resnets.ff_spatio_temp_resnet_3d import (
	FFSpatioTempResnetBlock3D, FFSpatioTempResDownsample3D, FFSpatioTempResUpsample3D
)
from .transformers.ff_spatio_temp_transformer_3d import FFSpatioTempTransformer3DModel
from .transformers.ff_spatio_audio_temp_transformer_3d import FFSpatioAudioTempTransformer3DModel


def create_custom_forward(module, return_dict=None):
	def custom_forward(*inputs):
		if return_dict is not None:
			return module(*inputs, return_dict=return_dict)
		else:
			return module(*inputs)
	
	return custom_forward


def get_down_block(
	down_block_type,
	num_layers,
	in_channels,
	out_channels,
	temb_channels,
	add_downsample,
	resnet_eps,
	resnet_act_fn,
	attn_num_head_channels,
	resnet_groups=None,
	cross_attention_dim=None,
	downsample_padding=None,
	dual_cross_attention=False,
	use_linear_projection=False,
	only_cross_attention=False,
	upcast_attention=False,
	resnet_time_scale_shift="default",
	audio_cross_attention_dim=None
):
	down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
	if down_block_type == "FFSpatioTempResDownBlock3D":
		return FFSpatioTempResDownBlock3D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			resnet_time_scale_shift=resnet_time_scale_shift
		)
	elif down_block_type == "FFSpatioTempCrossAttnDownBlock3D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
		return FFSpatioTempCrossAttnDownBlock3D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			cross_attention_dim=cross_attention_dim,
			attn_num_head_channels=attn_num_head_channels,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
			resnet_time_scale_shift=resnet_time_scale_shift
		)
	elif down_block_type == "FFSpatioAudioTempCrossAttnDownBlock3D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
		return FFSpatioAudioTempCrossAttnDownBlock3D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			cross_attention_dim=cross_attention_dim,
			audio_cross_attention_dim=audio_cross_attention_dim,
			attn_num_head_channels=attn_num_head_channels,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
			resnet_time_scale_shift=resnet_time_scale_shift	
		)
	raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
	up_block_type,
	num_layers,
	in_channels,
	out_channels,
	prev_output_channel,
	temb_channels,
	add_upsample,
	resnet_eps,
	resnet_act_fn,
	attn_num_head_channels,
	resnet_groups=None,
	cross_attention_dim=None,
	dual_cross_attention=False,
	use_linear_projection=False,
	only_cross_attention=False,
	upcast_attention=False,
	resnet_time_scale_shift="default",
	audio_cross_attention_dim=None
):
	up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
	if up_block_type == "FFSpatioTempResUpBlock3D":
		return FFSpatioTempResUpBlock3D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			resnet_time_scale_shift=resnet_time_scale_shift
		)
	elif up_block_type == "FFSpatioTempCrossAttnUpBlock3D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
		return FFSpatioTempCrossAttnUpBlock3D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			cross_attention_dim=cross_attention_dim,
			attn_num_head_channels=attn_num_head_channels,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
			resnet_time_scale_shift=resnet_time_scale_shift
		)
	elif up_block_type == "FFSpatioAudioTempCrossAttnUpBlock3D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
		return FFSpatioAudioTempCrossAttnUpBlock3D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			cross_attention_dim=cross_attention_dim,
			audio_cross_attention_dim=audio_cross_attention_dim,
			attn_num_head_channels=attn_num_head_channels,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
			resnet_time_scale_shift=resnet_time_scale_shift
		)
	raise ValueError(f"{up_block_type} does not exist.")


def get_mid_block(
	mid_block_type,
	in_channels,
	temb_channels,
	resnet_eps,
	resnet_act_fn,
	output_scale_factor,
	resnet_time_scale_shift,
	cross_attention_dim,
	attn_num_head_channels,
	resnet_groups,
	dual_cross_attention,
	use_linear_projection,
	upcast_attention,
	audio_cross_attention_dim=None
):
	if mid_block_type == "FFSpatioTempCrossAttnUNetMidBlock3D":
		return FFSpatioTempCrossAttnUNetMidBlock3D(
			in_channels=in_channels,
			temb_channels=temb_channels,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			output_scale_factor=output_scale_factor,
			resnet_time_scale_shift=resnet_time_scale_shift,
			cross_attention_dim=cross_attention_dim,
			attn_num_head_channels=attn_num_head_channels,
			resnet_groups=resnet_groups,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			upcast_attention=upcast_attention
		)
	elif mid_block_type == "FFSpatioAudioTempCrossAttnUNetMidBlock3D":
		return FFSpatioAudioTempCrossAttnUNetMidBlock3D(
			in_channels=in_channels,
			temb_channels=temb_channels,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			output_scale_factor=output_scale_factor,
			resnet_time_scale_shift=resnet_time_scale_shift,
			cross_attention_dim=cross_attention_dim,
			audio_cross_attention_dim=audio_cross_attention_dim,
			attn_num_head_channels=attn_num_head_channels,
			resnet_groups=resnet_groups,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			upcast_attention=upcast_attention
		)
	raise ValueError(f"{mid_block_type} does not exist.")


##### Image Condition Blocks #####

class FFSpatioTempResDownBlock3D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_time_scale_shift: str = "default",
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		resnet_pre_norm: bool = True,
		output_scale_factor=1.0,
		add_downsample=True,
		downsample_padding=1
	):
		super().__init__()
		resnets = []

		for i in range(num_layers):
			in_channels = in_channels if i == 0 else out_channels
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm
				)
			)

		self.resnets = nn.ModuleList(resnets)

		if add_downsample:
			self.downsamplers = nn.ModuleList(
				[
					FFSpatioTempResDownsample3D(
						out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
					)
				]
			)
		else:
			self.downsamplers = None

		self.gradient_checkpointing = False

	def forward(self, hidden_states, temb=None):
		output_states = ()

		for resnet in self.resnets:
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
			else:
				hidden_states = resnet(hidden_states, temb)

			output_states += (hidden_states,)

		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)

			output_states += (hidden_states,)

		return hidden_states, output_states


class FFSpatioTempResUpBlock3D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		prev_output_channel: int,
		out_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_time_scale_shift: str = "default",
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		resnet_pre_norm: bool = True,
		output_scale_factor=1.0,
		add_upsample=True
	):
		super().__init__()
		resnets = []

		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels

			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm
				)
			)

		self.resnets = nn.ModuleList(resnets)

		if add_upsample:
			self.upsamplers = nn.ModuleList([FFSpatioTempResUpsample3D(out_channels, use_conv=True, out_channels=out_channels)])
		else:
			self.upsamplers = None

		self.gradient_checkpointing = False

	def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
		for resnet in self.resnets:
			# pop res hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]
			hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
			else:
				hidden_states = resnet(hidden_states, temb)

		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states, upsample_size)

		return hidden_states


class FFSpatioTempCrossAttnUNetMidBlock3D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			temb_channels: int,
			dropout: float = 0.0,
			num_layers: int = 1,
			resnet_eps: float = 1e-6,
			resnet_time_scale_shift: str = "default",
			resnet_act_fn: str = "swish",
			resnet_groups: int = 32,
			resnet_pre_norm: bool = True,
			attn_num_head_channels=1,
			output_scale_factor=1.0,
			cross_attention_dim=1280,
			dual_cross_attention=False,
			use_linear_projection=False,
			upcast_attention=False
	):
		super().__init__()
		
		self.has_cross_attention = True
		self.attn_num_head_channels = attn_num_head_channels
		resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
		
		# there is always at least one resnet
		resnets = [
			FFSpatioTempResnetBlock3D(
				in_channels=in_channels,
				out_channels=in_channels,
				temb_channels=temb_channels,
				eps=resnet_eps,
				groups=resnet_groups,
				dropout=dropout,
				time_embedding_norm=resnet_time_scale_shift,
				non_linearity=resnet_act_fn,
				output_scale_factor=output_scale_factor,
				pre_norm=resnet_pre_norm
			)
		]
		attentions = []
		
		for _ in range(num_layers):
			if dual_cross_attention:
				raise NotImplementedError
			attentions.append(
				FFSpatioTempTransformer3DModel(
					attn_num_head_channels,
					in_channels // attn_num_head_channels,
					in_channels=in_channels,
					num_layers=1,
					cross_attention_dim=cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					upcast_attention=upcast_attention,
				)
			)
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=in_channels,
					out_channels=in_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
					
				)
			)
		
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)
		
		self.gradient_checkpointing = False
	
	def forward(self, hidden_states, temb=None, encoder_hidden_states=None,
	            cross_attention_kwargs=None):
		if self.training and self.gradient_checkpointing:
			hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.resnets[0]), hidden_states, temb)
		else:
			hidden_states = self.resnets[0](hidden_states, temb)
		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(
					create_custom_forward(attn, return_dict=False),
					hidden_states,
					encoder_hidden_states,
					cross_attention_kwargs
				)[0]
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
			else:
				hidden_states = attn(
					hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					cross_attention_kwargs=cross_attention_kwargs
				).sample
				hidden_states = resnet(hidden_states, temb)
		
		return hidden_states


class FFSpatioTempCrossAttnDownBlock3D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			temb_channels: int,
			dropout: float = 0.0,
			num_layers: int = 1,
			resnet_eps: float = 1e-6,
			resnet_time_scale_shift: str = "default",
			resnet_act_fn: str = "swish",
			resnet_groups: int = 32,
			resnet_pre_norm: bool = True,
			attn_num_head_channels=1,
			cross_attention_dim=1280,
			output_scale_factor=1.0,
			downsample_padding=1,
			add_downsample=True,
			dual_cross_attention=False,
			use_linear_projection=False,
			only_cross_attention=False,
			upcast_attention=False,
			
	):
		super().__init__()
		resnets = []
		attentions = []
		
		self.has_cross_attention = True
		self.attn_num_head_channels = attn_num_head_channels
		
		for i in range(num_layers):
			in_channels = in_channels if i == 0 else out_channels
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
					
				)
			)
			if dual_cross_attention:
				raise NotImplementedError
			attentions.append(
				FFSpatioTempTransformer3DModel(
					attn_num_head_channels,
					out_channels // attn_num_head_channels,
					in_channels=out_channels,
					num_layers=1,
					cross_attention_dim=cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					only_cross_attention=only_cross_attention,
					upcast_attention=upcast_attention,
				)
			)
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)
		
		if add_downsample:
			self.downsamplers = nn.ModuleList(
				[
					FFSpatioTempResDownsample3D(
						out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op",
						
					)
				]
			)
		else:
			self.downsamplers = None
		
		self.gradient_checkpointing = False
	
	def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None,
	            cross_attention_kwargs=None):
		output_states = ()
		
		for resnet, attn in zip(self.resnets, self.attentions):
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
				hidden_states = torch.utils.checkpoint.checkpoint(
					create_custom_forward(attn, return_dict=False),
					hidden_states,
					encoder_hidden_states,
					cross_attention_kwargs
				)[0]
			else:
				hidden_states = resnet(hidden_states, temb)
				hidden_states = attn(
					hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					cross_attention_kwargs=cross_attention_kwargs,
				).sample
			
			output_states += (hidden_states,)
		
		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)
			
			output_states += (hidden_states,)
		
		return hidden_states, output_states


class FFSpatioTempCrossAttnUpBlock3D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			prev_output_channel: int,
			temb_channels: int,
			dropout: float = 0.0,
			num_layers: int = 1,
			resnet_eps: float = 1e-6,
			resnet_time_scale_shift: str = "default",
			resnet_act_fn: str = "swish",
			resnet_groups: int = 32,
			resnet_pre_norm: bool = True,
			attn_num_head_channels=1,
			cross_attention_dim=1280,
			output_scale_factor=1.0,
			add_upsample=True,
			dual_cross_attention=False,
			use_linear_projection=False,
			only_cross_attention=False,
			upcast_attention=False,
			
	):
		super().__init__()
		resnets = []
		attentions = []
		
		self.has_cross_attention = True
		self.attn_num_head_channels = attn_num_head_channels
		
		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels
			
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
					
				)
			)
			if dual_cross_attention:
				raise NotImplementedError
			attentions.append(
				FFSpatioTempTransformer3DModel(
					attn_num_head_channels,
					out_channels // attn_num_head_channels,
					in_channels=out_channels,
					num_layers=1,
					cross_attention_dim=cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					only_cross_attention=only_cross_attention,
					upcast_attention=upcast_attention,
				)
			)
		
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)
		
		if add_upsample:
			self.upsamplers = nn.ModuleList(
				[FFSpatioTempResUpsample3D(out_channels, use_conv=True, out_channels=out_channels,
				                                 )])
		else:
			self.upsamplers = None
		
		self.gradient_checkpointing = False
	
	def forward(
			self,
			hidden_states,
			res_hidden_states_tuple,
			temb=None,
			encoder_hidden_states=None,
			upsample_size=None,
			attention_mask=None,
			cross_attention_kwargs=None
	):
		for resnet, attn in zip(self.resnets, self.attentions):
			# pop res hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]
			hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
			
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
				hidden_states = torch.utils.checkpoint.checkpoint(
					create_custom_forward(attn, return_dict=False),
					hidden_states,
					encoder_hidden_states,
					cross_attention_kwargs
				)[0]
			else:
				hidden_states = resnet(hidden_states, temb)
				hidden_states = attn(
					hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					cross_attention_kwargs=cross_attention_kwargs,
				).sample
		
		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states, upsample_size)
		
		return hidden_states


##### Audio Condition Blocks #####

class FFSpatioAudioTempCrossAttnUNetMidBlock3D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			temb_channels: int,
			dropout: float = 0.0,
			num_layers: int = 1,
			resnet_eps: float = 1e-6,
			resnet_time_scale_shift: str = "default",
			resnet_act_fn: str = "swish",
			resnet_groups: int = 32,
			resnet_pre_norm: bool = True,
			attn_num_head_channels=1,
			output_scale_factor=1.0,
			cross_attention_dim=1280,
			audio_cross_attention_dim=768,
			dual_cross_attention=False,
			use_linear_projection=False,
			upcast_attention=False,
			
	):
		super().__init__()
		
		self.has_cross_attention = True
		self.attn_num_head_channels = attn_num_head_channels
		resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
		
		# there is always at least one resnet
		resnets = [
			FFSpatioTempResnetBlock3D(
				in_channels=in_channels,
				out_channels=in_channels,
				temb_channels=temb_channels,
				eps=resnet_eps,
				groups=resnet_groups,
				dropout=dropout,
				time_embedding_norm=resnet_time_scale_shift,
				non_linearity=resnet_act_fn,
				output_scale_factor=output_scale_factor,
				pre_norm=resnet_pre_norm,
				
			)
		]
		attentions = []
		
		for _ in range(num_layers):
			if dual_cross_attention:
				raise NotImplementedError
			attentions.append(
				FFSpatioAudioTempTransformer3DModel(
					attn_num_head_channels,
					in_channels // attn_num_head_channels,
					in_channels=in_channels,
					num_layers=1,
					cross_attention_dim=cross_attention_dim,
					audio_cross_attention_dim=audio_cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					upcast_attention=upcast_attention,
				)
			)
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=in_channels,
					out_channels=in_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
					
				)
			)
		
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)
		
		self.gradient_checkpointing = False
	
	def forward(self, hidden_states, temb=None,
	            encoder_hidden_states=None, attention_mask=None,
	            audio_encoder_hidden_states=None, audio_attention_mask=None,
	            cross_attention_kwargs=None):
		assert cross_attention_kwargs is None
		if self.training and self.gradient_checkpointing:
			hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.resnets[0]), hidden_states,
			                                                  temb)
		else:
			hidden_states = self.resnets[0](hidden_states, temb)
		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(
					create_custom_forward(attn, return_dict=False),
					hidden_states,
					encoder_hidden_states,
					audio_encoder_hidden_states,
					audio_attention_mask,
				)[0]
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
			else:
				hidden_states = attn(
					hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					audio_encoder_hidden_states=audio_encoder_hidden_states,
					audio_attention_mask=audio_attention_mask,
					cross_attention_kwargs=cross_attention_kwargs
				).sample
				hidden_states = resnet(hidden_states, temb)
		
		return hidden_states


class FFSpatioAudioTempCrossAttnDownBlock3D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			temb_channels: int,
			dropout: float = 0.0,
			num_layers: int = 1,
			resnet_eps: float = 1e-6,
			resnet_time_scale_shift: str = "default",
			resnet_act_fn: str = "swish",
			resnet_groups: int = 32,
			resnet_pre_norm: bool = True,
			attn_num_head_channels=1,
			cross_attention_dim=1280,
			audio_cross_attention_dim=768,
			output_scale_factor=1.0,
			downsample_padding=1,
			add_downsample=True,
			dual_cross_attention=False,
			use_linear_projection=False,
			only_cross_attention=False,
			upcast_attention=False,
			
	):
		super().__init__()
		resnets = []
		attentions = []
		
		self.has_cross_attention = True
		self.attn_num_head_channels = attn_num_head_channels
		
		for i in range(num_layers):
			in_channels = in_channels if i == 0 else out_channels
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
					
				)
			)
			if dual_cross_attention:
				raise NotImplementedError
			attentions.append(
				FFSpatioAudioTempTransformer3DModel(
					attn_num_head_channels,
					out_channels // attn_num_head_channels,
					in_channels=out_channels,
					num_layers=1,
					cross_attention_dim=cross_attention_dim,
					audio_cross_attention_dim=audio_cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					only_cross_attention=only_cross_attention,
					upcast_attention=upcast_attention
				)
			)
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)
		
		if add_downsample:
			self.downsamplers = nn.ModuleList(
				[
					FFSpatioTempResDownsample3D(
						out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op",
						
					)
				]
			)
		else:
			self.downsamplers = None
		
		self.gradient_checkpointing = False
	
	def forward(self, hidden_states, temb=None,
	            encoder_hidden_states=None, attention_mask=None,
	            audio_encoder_hidden_states=None, audio_attention_mask=None,
	            cross_attention_kwargs=None):
		output_states = ()
		
		for resnet, attn in zip(self.resnets, self.attentions):
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
				hidden_states = torch.utils.checkpoint.checkpoint(
					create_custom_forward(attn, return_dict=False),
					hidden_states,
					encoder_hidden_states,
					audio_encoder_hidden_states,
					audio_attention_mask
				)[0]
			else:
				hidden_states = resnet(hidden_states, temb)
				hidden_states = attn(
					hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					audio_encoder_hidden_states=audio_encoder_hidden_states,
					audio_attention_mask=audio_attention_mask,
					cross_attention_kwargs=cross_attention_kwargs,
				).sample
			
			output_states += (hidden_states,)
		
		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)
			
			output_states += (hidden_states,)
		
		return hidden_states, output_states


class FFSpatioAudioTempCrossAttnUpBlock3D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			prev_output_channel: int,
			temb_channels: int,
			dropout: float = 0.0,
			num_layers: int = 1,
			resnet_eps: float = 1e-6,
			resnet_time_scale_shift: str = "default",
			resnet_act_fn: str = "swish",
			resnet_groups: int = 32,
			resnet_pre_norm: bool = True,
			attn_num_head_channels=1,
			cross_attention_dim=1280,
			audio_cross_attention_dim=768,
			output_scale_factor=1.0,
			add_upsample=True,
			dual_cross_attention=False,
			use_linear_projection=False,
			only_cross_attention=False,
			upcast_attention=False,
			
	):
		super().__init__()
		resnets = []
		attentions = []
		
		self.has_cross_attention = True
		self.attn_num_head_channels = attn_num_head_channels
		
		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels
			
			resnets.append(
				FFSpatioTempResnetBlock3D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
					
				)
			)
			if dual_cross_attention:
				raise NotImplementedError
			attentions.append(
				FFSpatioAudioTempTransformer3DModel(
					attn_num_head_channels,
					out_channels // attn_num_head_channels,
					in_channels=out_channels,
					num_layers=1,
					cross_attention_dim=cross_attention_dim,
					audio_cross_attention_dim=audio_cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					only_cross_attention=only_cross_attention,
					upcast_attention=upcast_attention,
				)
			)
		
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)
		
		if add_upsample:
			self.upsamplers = nn.ModuleList(
				[FFSpatioTempResUpsample3D(out_channels, use_conv=True, out_channels=out_channels,
				                                 )])
		else:
			self.upsamplers = None
		
		self.gradient_checkpointing = False
	
	def forward(
			self,
			hidden_states,
			res_hidden_states_tuple,
			temb=None,
			encoder_hidden_states=None,
			attention_mask=None,
			audio_encoder_hidden_states=None,
			audio_attention_mask=None,
			upsample_size=None,
			cross_attention_kwargs=None
	):
		assert cross_attention_kwargs is None
		for resnet, attn in zip(self.resnets, self.attentions):
			# pop res hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]
			hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
			
			if self.training and self.gradient_checkpointing:
				hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
				hidden_states = torch.utils.checkpoint.checkpoint(
					create_custom_forward(attn, return_dict=False),
					hidden_states,
					encoder_hidden_states,
					audio_encoder_hidden_states,
					audio_attention_mask,
					cross_attention_kwargs
				)[0]
			else:
				hidden_states = resnet(hidden_states, temb)
				hidden_states = attn(
					hidden_states,
					encoder_hidden_states=encoder_hidden_states,
					audio_encoder_hidden_states=audio_encoder_hidden_states,
					audio_attention_mask=audio_attention_mask,
					cross_attention_kwargs=cross_attention_kwargs,
				).sample
		
		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states, upsample_size)
		
		return hidden_states


all_modules = [
	##### Image Condition #####
	
	FFSpatioTempResDownBlock3D,
	FFSpatioTempResUpBlock3D,
	
	FFSpatioTempCrossAttnUNetMidBlock3D,
	FFSpatioTempCrossAttnDownBlock3D,
	FFSpatioTempCrossAttnUpBlock3D,
	
	##### Audio Condition #####
	
	FFSpatioAudioTempCrossAttnUNetMidBlock3D,
	FFSpatioAudioTempCrossAttnDownBlock3D,
	FFSpatioAudioTempCrossAttnUpBlock3D,
]