import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class AVSyncContrastiveTrainer(nn.Module):
	
	def __init__(
			self,
			audio_encoder,
			video_encoder,
			head,
			tau=1.0
	):
		super().__init__()
		self.audio_encoder = audio_encoder
		self.video_encoder = video_encoder
		self.head = head
		self.tau = tau
	
	def forward(self, audios, videos):
		
		device = audios.device
		batchsize, num_pairs = audios.shape[:2]
		
		audios = rearrange(audios, "b k c n t -> (b k) c n t")
		videos = rearrange(videos, "b k c f h w -> (b k) c f h w")
		
		audio_embeddings = self.audio_encoder(audios) # (bk c)
		video_embeddings = self.video_encoder(videos) # (bk c)
		
		predictions = self.head(
			repeat(audio_embeddings, "(b p) c -> (b p q) c", p=num_pairs, q=num_pairs),
			repeat(video_embeddings, "(b q) c -> (b p q) c", p=num_pairs, q=num_pairs)
		)[:, 0].contiguous() # (bpq, )
		
		labels = repeat(torch.arange(num_pairs).long(), "p -> (b p)", b=batchsize, p=num_pairs).to(device)
		
		av_loss = F.cross_entropy(
			rearrange(predictions, "(b p q) -> (b p) q", b=batchsize, p=num_pairs, q=num_pairs) / self.tau, labels
		)
		va_loss = F.cross_entropy(
			rearrange(predictions, "(b p q) -> (b q) p", b=batchsize, p=num_pairs, q=num_pairs) / self.tau, labels
		)
		
		av_logits = rearrange(predictions, "(b p q) -> (b p) q", p=num_pairs, q=num_pairs)
		av_acc = (av_logits.argmax(dim=1) == labels).float().mean()
		
		va_logits = rearrange(predictions, "(b p q) -> (b q) p", p=num_pairs, q=num_pairs)
		va_acc = (va_logits.argmax(dim=1) == labels).float().mean()

		return av_loss, va_loss, av_acc, va_acc

	def save_pretrained(self, save_dir):
		self.audio_encoder.save_pretrained(os.path.join(save_dir, "audio_encoder"))
		self.video_encoder.save_pretrained(os.path.join(save_dir, "video_encoder"))
		self.head.save_pretrained(os.path.join(save_dir, "head"))
		
		