import os
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models import AutoencoderKL

from avgen.models.audio_encoders import ImageBindSegmaskAudioEncoder
from avgen.models.unets import AudioUNet3DConditionModel


def prob_mask_like(shape, prob, device):
	if prob == 1:
		return torch.ones(shape, device = device, dtype = torch.bool)
	elif prob == 0:
		return torch.zeros(shape, device = device, dtype = torch.bool)
	else:
		return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


class AudioCondAnimationTrainer(nn.Module):
	def __init__(
			self,
			vae: AutoencoderKL,
			audio_encoder: ImageBindSegmaskAudioEncoder,
			unet: AudioUNet3DConditionModel,
			scheduler: KarrasDiffusionSchedulers,
			text_cond_drop_prob: float = 0.0,
			audio_cond_drop_prob: float = 0.0,
			loss_on_first_frame: bool = False,
	):
		super().__init__()
		self.audio_encoder = audio_encoder
		self.vae = vae
		self.unet = unet
		self.scheduler = scheduler
		
		self.text_cond_drop_prob = text_cond_drop_prob
		self.audio_cond_drop_prob = audio_cond_drop_prob
		self.loss_on_first_frame = loss_on_first_frame
		
		self.null_text_encoding = nn.Parameter(torch.zeros(1, 1, 77, 768), requires_grad=False)
		encoding = torch.load("pretrained/openai-clip-l_null_text_encoding.pt", map_location="cpu").view(self.null_text_encoding.shape)
		self.null_text_encoding.copy_(encoding)
	
	@property
	def device(self):
		return self.unet.device
	
	@property
	def dtype(self):
		return self.unet.dtype
	
	@torch.no_grad()
	def compress_to_latents(self, images: torch.Tensor):
		"""
		Input:
			@videos: (b 3 H W), normalized by mean and std
		Output:
			@image_embeds: (b c h w)
		"""
		latents = self.vae.encode(images).latent_dist.sample()
		# mult by scaling factor for diffusion pass
		latents = latents * self.vae.config.scaling_factor
		return latents
	
	def forward(
			self,
			videos: torch.Tensor,
			audios: torch.Tensor,
			text_encodings: torch.Tensor
	):
		assert videos.ndim == 5, "videos must be in shape (b c f h w) in range [0, 1]"
		batchsize, video_length = videos.size(0), videos.size(2)
		
		# 1. Encode videos into latents
		#    Encode audio into encodings and masks
		with torch.no_grad():
			videos = rearrange(videos, "b c f h w -> (b f) c h w")
			videos = (videos-0.5)/0.5
			video_latents = self.compress_to_latents(videos)
			video_latents = rearrange(video_latents, "(b f) c h w -> b c f h w", f=video_length)

			_, audio_encodings, audio_masks = self.audio_encoder(audios, return_dict=False)
			_, null_audio_encodings, null_audio_masks = self.audio_encoder(torch.zeros_like(audios), return_dict=False)
		
		# 2. Prepare null image, text, and audio encoding
		if self.training:
			text_condition_keep_mask = prob_mask_like((batchsize,), 1 - self.text_cond_drop_prob, device=self.device)
			audio_condition_keep_mask = prob_mask_like((batchsize,), 1 - self.audio_cond_drop_prob, device=self.device)
		else:
			text_condition_keep_mask = prob_mask_like((batchsize,), 1., device=self.device)
			audio_condition_keep_mask = prob_mask_like((batchsize,), 1., device=self.device)
		
		text_encodings = text_encodings.to(self.dtype)
		text_encodings = torch.where(
			rearrange(text_condition_keep_mask, 'b -> b 1 1'),
			text_encodings,
			self.null_text_encoding[0].expand_as(text_encodings).contiguous()
		)
		text_encodings = text_encodings.unsqueeze(1).expand(batchsize, video_length, *self.null_text_encoding.shape[2:]).contiguous()
		
		audio_encodings = torch.where(
			rearrange(audio_condition_keep_mask, "b -> b 1 1"),
			audio_encodings,
			null_audio_encodings
		)
		audio_encodings = repeat(audio_encodings, "b n c-> b f n c", f=video_length).contiguous().to(self.dtype)
		audio_masks = torch.where(
			rearrange(audio_condition_keep_mask, "b -> b 1 1"),
			audio_masks,
			null_audio_masks,
		) # (b f n)
		
		# 5. diffusion schedule
		timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batchsize,)).long().to(device=self.device)
		noise = torch.randn_like(video_latents)
		noisy_video_latents = self.scheduler.add_noise(video_latents, noise, timesteps)
		
		first_frame_latents = video_latents[:, :, 0:1].contiguous()
		noisy_video_latents = torch.cat([
			first_frame_latents,
			noisy_video_latents[:, :, 1:]
		], dim=2).contiguous()
			
		if self.scheduler.config.prediction_type == "epsilon":
			target = noise
		elif self.scheduler.config.prediction_type == "v_prediction":
			target = self.scheduler.get_velocity(video_latents, noise, timesteps)
		else:
			raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
		
		predicted_latents = self.unet(
			sample=noisy_video_latents,
			timestep=timesteps,
			encoder_hidden_states=text_encodings,
			audio_encoder_hidden_states=audio_encodings,
			audio_attention_mask=audio_masks,
			return_dict=True
		).sample
		
		if self.loss_on_first_frame:
			loss = F.mse_loss(predicted_latents.float(), target.float())
		else:
			loss = F.mse_loss(predicted_latents[:, :, 1:].float(), target[:, :, 1:].float())
		
		return loss
	
	def save_pretrained(self, directory):
		os.makedirs(directory, exist_ok=True)
		self.unet.save_pretrained(os.path.join(directory, "unet"))
		self.audio_encoder.save_pretrained(os.path.join(directory, "audio_encoder"))
		